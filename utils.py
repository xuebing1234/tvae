from SinkhornDistance import SinkhornDistance
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
import math
import numpy as np
np.seterr(all='raise')
import torch
import torch.distributions as td
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from tqdm import tqdm,trange
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, average_precision_score


class DisentangledVAE:
    def __init__(
        self, input_dimension, latent_dimension, hidden_layer_width=1000, n_epochs=100, 
        number_of_labels=3, weight=[3,3,3], device=None
    ):
        self.n_epochs=n_epochs
        self.hidden_layer_width = hidden_layer_width
        self.input_dimension = input_dimension
        self.latent_dimension = latent_dimension
        self.number_of_labels = number_of_labels #supervised dimension
        self.pred_weight = weight
        self.beta = 1
        self.recon_weight = 1
        self.KL_weight=1
        self.z_var = 1
        self.reg_weight = 1
        self.wasserstein=1
        
        self.decoder = nn.Sequential(
            torch.nn.Linear(self.latent_dimension, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, self.input_dimension),
        ).to(device)
        self.encoder = nn.Sequential(
            torch.nn.Linear(self.input_dimension, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, self.hidden_layer_width),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_layer_width, (2 * self.latent_dimension)),
        ).to(device)
        self.optimizer = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=1e-4
        )
        self.mse = nn.MSELoss(reduction='mean')
        self.cn_loss = nn.BCEWithLogitsLoss(reduction='mean')
        self.device=device
        self.batch_size = 64
        self.early_stopper = EarlyStopper(patience=3, min_delta=0.05)

        self.generate_data=False
        self.kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
        self.K=10
        self.sinkhorn = SinkhornDistance(eps=0.1, 
                                         max_iter=100, 
                                         device= device, 
                                         reduction=None)
        self.plot=False
        #deep fake loss
        self.sample_ratio=.1
        self.generate_data=True
        


    def weights_init(self, layer):
        if type(layer) == nn.Linear:
            torch.nn.init.orthogonal_(layer.weight)

    def matrix_log_density_gaussian(self, x, mu, logvar):
        # broadcast to get probability of x given any instance(row) in (mu,logvar)
        # [k,:,:] : probability of kth row in x from all rows in (mu,logvar)
        x = x.view(self.batch_size, 1, self.latent_dimension)
        mu = mu.view(1, self.batch_size, self.latent_dimension)
        logvar = logvar.view(1, self.batch_size, self.latent_dimension)
        return td.Normal(loc=mu, scale=(torch.exp(logvar)) ** 0.5).log_prob(x)

    def log_importance_weight_matrix(self):
        """
        Calculates a log importance weight matrix
        Parameters
        ----------
        batch_size: int
            number of training images in the batch
        dataset_size: int
        number of training images in the dataset
        """
        N = self.n_data
        M = self.batch_size - 1
        strat_weight = (N - M) / (N * M)
        W = torch.Tensor(self.batch_size, self.batch_size).fill_(1 / M)
        W.view(-1)[:: M + 1] = 1 / N
        W.view(-1)[1 :: M + 1] = strat_weight
        W[M - 1, 0] = strat_weight
        return W.log()

    def get_log_qz_prodzi(self, latent_sample, latent_dist, is_mss=True):
        mat_log_qz = self.matrix_log_density_gaussian(
            latent_sample,
            latent_dist[..., : self.latent_dimension],
            latent_dist[..., self.latent_dimension :],
        )
        if is_mss:
            # use stratification
            log_iw_mat = self.log_importance_weight_matrix().to(
                latent_sample.device
            )
            mat_log_qz = mat_log_qz + log_iw_mat.view(self.batch_size, self.batch_size, 1)
            log_qz = torch.logsumexp(
                log_iw_mat + mat_log_qz.sum(2), dim=1, keepdim=False
            )
            log_prod_qzi = torch.logsumexp(
                log_iw_mat.view(self.batch_size, self.batch_size, 1) + mat_log_qz,
                dim=1,
                keepdim=False,
            ).sum(1)

        else:
            log_prod_qzi = (
                torch.logsumexp(
                    mat_log_qz, dim=1, keepdim=False
                )  # sum of probabilities in each latent dimension
                - math.log(self.batch_size * self.n_data)
            ).sum(1)
            log_qz = torch.logsumexp(
                mat_log_qz.sum(2),  # sum of probabilities across all latent dimensions
                dim=1,
                keepdim=False,
            ) - math.log(self.batch_size * self.n_data)

        return log_qz, log_prod_qzi

    def _kl_normal_loss(self, mean, logvar):
        """
        Calculates the KL divergence between a normal distribution
        with diagonal covariance and a unit normal distribution.
        Parameters
        ----------
        mean : torch.Tensor
            Mean of the normal distribution. Shape (batch_size, latent_dim) where
            D is dimension of distribution.
        logvar : torch.Tensor
            Diagonal log variance of the normal distribution. Shape (batch_size,
            latent_dim)
        storer : dict
            Dictionary in which to store important variables for vizualisation.
        """
        latent_dim = mean.size(1)
        # batch mean of kl for each latent dimension
        latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
        total_kl = latent_kl.sum()

        return total_kl
    def generate_fake(self, x1,sample_ratio):
        out_encoder = self.encoder(x1.to(self.device))
        treatment = td.Independent(td.Normal(loc=out_encoder[:, :self.latent_dimension],
                                            scale=torch.exp(out_encoder[:, self.latent_dimension :]) ** 0.5,
                                            )
                                   ,1)
        x2 = treatment.rsample([sample_ratio]).reshape(
            (-1,self.latent_dimension)
        )
        return self.decoder(x2).cpu()

    def new_data(self,xhat_0,yhat_0,mask_0) :
        if yhat_0[:,0].sum()*2>len(yhat_0):
            minority = xhat_0[yhat_0[:,0]==0]
            minority_label = yhat_0[yhat_0[:,0]==0]
            minority_mask = mask_0[yhat_0[:,0]==0]
        else:
            minority = xhat_0[yhat_0[:,0]==1]
            minority_label = yhat_0[yhat_0[:,0]==1]
            minority_mask = mask_0[yhat_0[:,0]==1]
        number_of_new_samples = int(self.sample_ratio*(self.n_data-len(minority)))
        negative_sample_ratio = int(number_of_new_samples/len(minority)+0.5)

        upsampled_data=[]
        upsampled_label=[]
        upsampled_mask=[]
        for i,sample_ind in enumerate(np.array_split(np.arange(len(minority)),100)):
            upsampled_data.append(self.generate_fake(minority[sample_ind,:],negative_sample_ratio))
            upsampled_label.append(minority_label[sample_ind,:].repeat(negative_sample_ratio,1))
            upsampled_mask.append(minority_mask[sample_ind,:].repeat(negative_sample_ratio,1))
        upsampled_data=torch.cat(upsampled_data,dim=0)
        upsampled_label=torch.cat(upsampled_label,dim=0)
        upsampled_mask=torch.cat(upsampled_mask,dim=0)
        epoch_data = torch.cat([xhat_0, upsampled_data[:number_of_new_samples,:]], dim=0)
        epoch_label = torch.cat([yhat_0, upsampled_label[:number_of_new_samples,:]], dim=0)
        epoch_mask = torch.cat([mask_0, upsampled_mask[:number_of_new_samples,:]], dim=0)
        return epoch_data,epoch_label,epoch_mask    
    def pred_loss(self,targets,out_encoder):
        # when classification: cn_loss = nn.BCEWithLogitsLoss().cuda()
        pred_losses = [self.cn_loss(out_encoder[:,0].reshape(-1,1),
                                    targets[:,0].reshape(-1,1))]      
        if len(targets[targets[:,0]==1,1])>0:
            loc = out_encoder[targets[:,0]==1,1].reshape((-1,1))
            truth = targets[targets[:,0]==1,1].reshape((-1,1))
            pred_losses.append(self.cn_loss(loc,truth))
        else:
            pred_losses.append(torch.tensor([float('0')]).to(self.device))
        if len(targets[targets[:,0]==0,2])>0:
            loc = out_encoder[targets[:,0]==0,2].reshape((-1,1))
            truth = targets[targets[:,0]==0,2].reshape((-1,1))
            pred_losses.append(self.cn_loss(loc,truth))
        else:
            pred_losses.append(torch.tensor([float('0')]).to(self.device))
        return pred_losses
    def compute_mmd(self,z1, z2, reg_weight):
        prior_z__kernel = self.compute_kernel(z1, z1)
        z__kernel = self.compute_kernel(z2, z2)
        priorz_z__kernel = self.compute_kernel(z1, z2)

        mmd = reg_weight * prior_z__kernel.mean() + \
              reg_weight * z__kernel.mean() - \
              2 * reg_weight * priorz_z__kernel.mean()
        return mmd
    def compute_kernel(self,x1,x2,kernel_type='rbf'):
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)
        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)
        if kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')
        return result

    def compute_rbf(self,x1,x2,eps = 1e-7):
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.z_var
        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self,x1,x2,eps= 1e-7):
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by
                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))
        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()
        return result
    def _trainer(self, train_data, targets,mask):
        train_data=train_data.view(-1, self.input_dimension)
        targets = targets.view(-1,self.number_of_labels)
        if not self.batch_size == train_data.shape[0]:
            self.batch_size == train_data.shape[0]
        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        self.encoder.zero_grad()
        self.decoder.zero_grad()
        [
            loss,
            pred_loss,
            recon_loss,
            original_KL,
            tc_loss,
            mmd_loss,
        ] = self.compute_loss(data=train_data,targets = targets,mask=mask)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters())
        #                               ,self.clip)
        self.optimizer.step()
        # print(loss.item(),
        #     pred_loss,
        #     recon_loss,
        #     original_KL,
        #     tc_loss,
        #     mmd_loss)
        return (
            loss.item(),
            pred_loss,
            recon_loss,
            original_KL,
            tc_loss,
            mmd_loss
        )
    def trainer(self,train_data,test_data,train_label,test_label,train_mask, eval_data, eval_label
                ):
        self.encoder.apply(self.weights_init)
        self.decoder.apply(self.weights_init)
        train_loss = []
        val_loss = []
        test_score=-1
        
        train_data = torch.Tensor(train_data)
        train_mask = torch.Tensor(train_mask)
        test_data = torch.Tensor(test_data).to(self.device)
        eval_data = torch.Tensor(eval_data).to(self.device)
        
        train_label = torch.Tensor(train_label)
        # test_label = torch.Tensor(test_label).to(self.device)
        eval_label = torch.Tensor(eval_label).to(self.device)

        epoch_label = train_label.detach().clone()
        epoch_data = train_data.detach().clone()
        epoch_mask = train_mask.detach().clone()
        for epoch in trange(self.n_epochs, desc='epochs',leave=False):
            epoch_train_loss=[]
            train_set = torch.utils.data.TensorDataset(epoch_data, epoch_mask, epoch_label)
            train_loader = DataLoader(train_set, shuffle=True,
                                      num_workers=1, drop_last=True, 
                                      batch_size=self.batch_size)
            self.n_data = len(epoch_data)
            for i, batch_data in enumerate(train_loader,0):
                data,mask,label = batch_data
                (loss, pred_loss, 
                 recon_loss,original_KL,
                 tc_loss, mmd_loss) = self._trainer(data.to(self.device),
                                                  label.to(self.device),
                                                 mask.to(self.device))
                epoch_train_loss.append([loss, pred_loss,recon_loss,original_KL,
                                         tc_loss, mmd_loss])
            
            epoch_pred = self.encoder(eval_data)            
            with torch.no_grad():
                prediction_losses=self.pred_loss(eval_label ,
                                    epoch_pred
                                   )
                loss_prediction=  prediction_losses[0]*self.pred_weight[0] + \
        prediction_losses[1]*self.pred_weight[1] + \
        prediction_losses[2]*self.pred_weight[2]
            train_loss.append(np.mean(epoch_train_loss,axis=0))
            # print('train', epoch,epoch_label.nanmean(axis=0),self.n_data)
            val_loss.append(loss_prediction)   
            if self.early_stopper.early_stop(loss_prediction):
                    aurocs=[]
                    with torch.no_grad():
                        encoded_test = self.encoder(test_data).cpu().numpy()
                    outcome = np.where(test_label[:,0]==1, test_label[:,1],test_label[:,2])
                    pred_outcome = np.where(test_label[:,0]==1, encoded_test[:,1],encoded_test[:,2])              
                    aurocs.append(roc_auc_score(test_label[:,0], encoded_test[:, 0]))
                    aurocs.append(roc_auc_score(outcome,pred_outcome))
                    # aurocs.append(roc_auc_score(test_label[test_label[:,0]==0,2], 
                    #                                 encoded_test[test_label[:,0]==0, 2]))
                    auprcs=[]
                    auprcs.append(average_precision_score(test_label[:,0], encoded_test[:, 0]))
                    auprcs.append(average_precision_score(outcome,pred_outcome))
                    # auprcs.append(average_precision_score(test_label[test_label[:,0]==1,1], 
                    #                                 encoded_test[test_label[:,0]==1, 1]))
                    # auprcs.append(average_precision_score(test_label[test_label[:,0]==0,2], 
                    #                                 encoded_test[test_label[:,0]==0, 2]))
                    test_score = [aurocs,auprcs]
                    break

            if self.generate_data:
                with torch.no_grad():
                    epoch_data,epoch_label,epoch_mask = self.new_data(train_data.detach().clone(),
                                                           train_label.detach().clone(),
                                                           train_mask.detach().clone())   
        if self.plot:
            f = plt.figure(figsize=(10,5))
            train_losses= np.array(train_loss).reshape(-1,6)
            val_loss = np.array(val_loss).reshape(-1,3)
            for i, loss in enumerate(['combined_loss','pred_loss','recon_loss','original KL',
                                      'tc_loss','mmd_loss']):
                train = train_losses[:,i]
                ax = f.add_subplot(1,1,1)
                plt.title(loss)
                plt.plot(train,label='train')
                if i ==1:
                    plt.plot(val_loss[:,0],label='eval_assignment')
                    plt.plot(val_loss[:,1],label='eval_treatment')
                    plt.plot(val_loss[:,2],label='eval_control')
                plt.legend()
                plt.show()
        return train_loss,val_loss,test_score
    def simple_mmd_loss(self,X_treat, X_control):
        """Calculate Maximum Mean Discrepancy loss."""
        return 2 * torch.norm(X_treat.mean(axis=0) - X_control.mean(axis=0))

    def compute_loss(self, data,targets,mask):
        out_encoder = self.encoder(data)
        #resample latent variables 
        q_zgivenxobs = td.Independent(
            td.Normal(
                loc=out_encoder[..., : self.latent_dimension],
                scale=torch.exp(out_encoder[..., self.latent_dimension :]) ** 0.5,
            ),
            1,
        )  # each row is a latent vector
        zgivenx_flat = q_zgivenxobs.rsample()
        zgivenx = zgivenx_flat.reshape((-1, self.latent_dimension))

        # calculate reconstruction loss
        out_decoder = self.decoder(zgivenx)
        recon_loss = self.mse(out_decoder[~mask.bool()], data[~mask.bool()])
        
        #calculate mmd_loss
        if len(targets[targets[:,0]==1,:])==0 or len(targets[targets[:,0]==1,:])==len(targets):
            mmd_loss = torch.tensor([float('0')]).to(self.device)
        else:
            control = td.Independent(td.Normal(loc=out_encoder[targets[:,0]==0, 
                                                               3:self.latent_dimension],
                                                scale=torch.exp(
                                                    out_encoder[targets[:,0]==0,
                                                                self.latent_dimension+3 :]) ** 0.5,
                                                )
                                       ,1)
            #x1 shape: (K*bs, 1, latent_dim)
            x1 = control.rsample([self.K]).view(-1,1,
                                           self.latent_dimension-self.number_of_labels)
            
            target = control.log_prob(x1).mean(axis=1).view(-1,1) 
            #target shape: (K*bs, control.size(0))
            treatment = td.Independent(td.Normal(loc=out_encoder[targets[:,0]==1, 
                                                                 3:self.latent_dimension],
                                                scale=torch.exp(
                                                    out_encoder[targets[:,0]==1, 
                                                                self.latent_dimension+3 :]) ** 0.5,
                                                )
                                       ,1)
            #prob_treatment shape: (K*bs, treatment.size(0))
            prob_treatment = treatment.log_prob(x1).mean(axis=1).view(-1,1)
            if self.wasserstein==1: 
                bias_corr = self.batch_size *  (self.batch_size - 1)
                reg_weight = self.reg_weight / bias_corr
                mmd_loss = 10**6*self.compute_mmd(target, 
                                            prob_treatment,
                                            reg_weight)
            elif self.wasserstein==2: 
                mmd_loss=10*self.simple_mmd_loss(target,
                                              prob_treatment)
                
            elif self.wasserstein==3:
                mmd_loss, _,_ = self.sinkhorn(target, 
                                              prob_treatment)
                mmd_loss = 10*mmd_loss
            elif self.wasserstein==4: 
                mmd_loss = 10**8*self.kl_loss(F.log_softmax(torch.exp(prob_treatment),dim=0),
                                        F.log_softmax(torch.exp(target),dim=0)
                                        )
            else:
                mmd_loss = torch.tensor([float('0')]).to(self.device)
        # calculate the original KL in VAE
        original_KL = self._kl_normal_loss(
            out_encoder[..., self.number_of_labels: self.latent_dimension],
            out_encoder[..., self.latent_dimension +self.number_of_labels :],
        )
        # prob of z given observations x
        log_pz = (
            td.Independent(
                td.Normal(
                    loc=torch.zeros_like(zgivenx), scale=torch.ones_like(zgivenx)
                ),
                1,
            )
            .log_prob(zgivenx)
            .mean()
        )
        log_q_zCx = q_zgivenxobs.log_prob(zgivenx).mean()

        log_qz, log_prod_qzi = self.get_log_qz_prodzi(
            zgivenx, out_encoder
        )
        # I[z;x] = KL[q(z,x)||q(x)q(z)] = E_x[KL[q(z|x)||q(z)]]
        mi_loss = (log_q_zCx - log_qz).mean()
        # TC[z] = KL[q(z)||\prod_i z_i]
        tc_loss = (log_qz - log_prod_qzi).mean()
        # dw_kl_loss is KL[q(z)||p(z)] instead of usual KL[q(z|x)||p(z))]
        dw_kl_loss = (log_prod_qzi - log_pz).mean()
        
        prediction_losses = self.pred_loss(targets,out_encoder)
        
        loss_prediction=  prediction_losses[0]*self.pred_weight[0] + \
        prediction_losses[1]*self.pred_weight[1] + \
        prediction_losses[2]*self.pred_weight[2]

        neg_bound =  loss_prediction+ recon_loss*self.recon_weight + original_KL*self.KL_weight \
        + tc_loss * self.beta + mmd_loss*self.gamma

        return (
            neg_bound,
            loss_prediction.item(),
            recon_loss.item()*self.recon_weight,
            original_KL.item()*self.KL_weight,
            tc_loss.item()* self.beta,
            mmd_loss.item()*self.gamma
        )

        