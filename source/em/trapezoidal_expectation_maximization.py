from transforms.trapezoidal_transform_function import TrapezoidalTransformFunction
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from em.expectation_maximization import ExpectationMaximization
from em.embody import _em_step_body_, _em_step_body, _em_step_body_row

class TrapezoidalExpectationMaximization(ExpectationMaximization):
    def __init__(self, cont_indices, ord_indices, window_size,window_width, sigma_init=None):
        self.transform_function = TrapezoidalTransformFunction(cont_indices, ord_indices, window_size=window_size,window_width=window_width)
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        # we assume boolean array of indices
        p = len(cont_indices)
        # By default, sigma corresponds to the correlation matrix of the permuted dataset (ordinals appear first, then continuous)
        if sigma_init is not None:
            self.sigma = sigma_init
        else:
            self.sigma = np.identity(p)
        # track what iteration the algorithm is on for use in weighting samples
        self.iteration = 1

    def partial_fit_and_predict(self, X_batch,cont_indices,ord_indices, max_workers=4, num_ord_updates=2, decay_coef=0.5, sigma_update=True, marginal_update = True, sigma_out=False):
        """
        Updates the fit of the copula using the data in X_batch and returns the 
        imputed values and the new correlation for the copula

        Args:
            X_batch (matrix): data matrix with entries to use to update copula and be imputed
            max_workers (positive int): the maximum number of workers for parallelism 
            num_ord_updates (positive int): the number of times to re-estimate the latent ordinals per batch
            decay_coef (float in (0,1)): tunes how much to weight new covariance estimates
        Returns:
            X_imp (matrix): X_batch with missing values imputed
        """
        self.cont_indices = cont_indices
        self.ord_indices = ord_indices
        #if not update:
            #old_window = self.transform_function.window
            #old_update_pos = self.transform_function.update_pos
        if marginal_update:
            self.transform_function.partial_fit(X_batch,cont_indices,ord_indices)
        # update marginals with the new batch
        #self.transform_function.partial_fit(X_batch)
        # print("X_batch", X_batch)
        res = self._fit_covariance(X_batch, max_workers, num_ord_updates, decay_coef, sigma_update, sigma_out)
        if sigma_out:
            Z_batch_imp, sigma = res
        else:
            Z_batch_imp = res
        # Rearrange Z_imp so that it's columns correspond to the columns of X
        Z_imp_rearranged = np.empty(X_batch.shape)
        Z_imp_rearranged[:,self.ord_indices] = Z_batch_imp[:,:np.sum(self.ord_indices)]
        Z_imp_rearranged[:,self.cont_indices] = Z_batch_imp[:,np.sum(self.ord_indices):]

        X_imp = np.empty(X_batch.shape)
        X_imp[:,self.cont_indices] = self.transform_function.partial_evaluate_cont_observed(Z_imp_rearranged, X_batch)
        X_imp[:,self.ord_indices] = self.transform_function.partial_evaluate_ord_observed(Z_imp_rearranged, X_batch)
        #if not update:
            #self.transform_function.window = old_window
            #self.transform_function.update_pos = old_update_pos 
         #   pass
        if sigma_out:
            return Z_imp_rearranged,X_imp, sigma
        else:
            return Z_imp_rearranged,X_imp

    def _fit_covariance(self, X_batch, max_workers=4, num_ord_updates=2, decay_coef=0.5, update=True, sigma_out=False, seed=1):
        """
        Updates the covariance matrix of the gaussian copula using the data 
        in X_batch and returns the imputed latent values corresponding to 
        entries of X_batch and the new sigma

        Args:
            X_batch (matrix): data matrix with which to update copula and with entries to be imputed
            max_workers: the maximum number of workers for parallelism 
            num_ord_updates: the number of times to restimate the latent ordinals per batch
            decay_coef (float in (0,1)): tunes how much to weight new covariance estimates
        Returns:
            sigma (matrix): an updated estimate of the covariance of the copula
            Z_imp (matrix): estimates of latent values in X_batch
        """
        Z_ord_lower, Z_ord_upper = self.transform_function.partial_evaluate_ord_latent(X_batch)
        Z_ord = self._init_Z_ord(Z_ord_lower, Z_ord_upper, seed)
        Z_cont = self.transform_function.partial_evaluate_cont_latent(X_batch)
        # Latent variable matrix with columns sorted as ordinal, continuous
        Z = np.concatenate((Z_ord, Z_cont), axis=1)
        batch_size, p = Z.shape
        # track previous sigma for the purpose of early stopping
        prev_sigma = self.sigma
        s=prev_sigma.shape[0]
        if p>s:
            temp=np.zeros([s,p-s])
            prev_sigma=np.insert(prev_sigma,[s],temp,axis=1)
            temp=np.zeros([p-s,p])
            prev_sigma=np.insert(prev_sigma,[s],temp,axis=0)
            for i in range(s,p):
                prev_sigma[i][i]=1.0

        Z_imp = np.zeros((batch_size, p))
        C = np.zeros((p, p))
        if max_workers==1:
            C, Z_imp, Z = _em_step_body(Z,Z_ord_lower, Z_ord_upper, prev_sigma, num_ord_updates)
        else:
            divide = batch_size/max_workers * np.arange(max_workers+1)
            divide = divide.astype(int)
            args = [(Z[divide[i]:divide[i+1],:],Z_ord_lower[divide[i]:divide[i+1],:], Z_ord_upper[divide[i]:divide[i+1],:], prev_sigma, num_ord_updates) for i in range(max_workers)]
            # divide each batch into max_workers parts instead of n parts
            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                res = pool.map(_em_step_body_, args)
                for i,(C_divide, Z_imp_divide, Z_divide) in enumerate(res):
                    Z_imp[divide[i]:divide[i+1],:] = Z_imp_divide
                    Z[divide[i]:divide[i+1],:] = Z_divide # not necessary if we only do on EM iteration 
                    C += C_divide
        C = C/batch_size
        sigma = np.cov(Z_imp, rowvar=False) + C
        sigma = self._project_to_correlation(sigma)

        if update:
            self.sigma = sigma*decay_coef + (1 - decay_coef)*prev_sigma
            prev_sigma = self.sigma
            self.iteration += 1
        if sigma_out:
            if update:
                sigma = self.get_sigma()
            else:
                sigma = self.get_sigma(sigma*decay_coef + (1 - decay_coef)*prev_sigma)
            return Z_imp, sigma
        else:
            return Z_imp

    def get_sigma(self, sigma=None):
        if sigma is None:
            sigma = self.sigma
        sigma_rearranged = np.empty(sigma.shape)
        sigma_rearranged[np.ix_(self.ord_indices,self.ord_indices)] = sigma[:np.sum(self.ord_indices),:np.sum(self.ord_indices)]
        sigma_rearranged[np.ix_(self.cont_indices,self.cont_indices)] = sigma[np.sum(self.ord_indices):,np.sum(self.ord_indices):]
        sigma_rearranged[np.ix_(self.cont_indices,self.ord_indices)] = sigma[np.sum(self.ord_indices):,:np.sum(self.ord_indices)]
        sigma_rearranged[np.ix_(self.ord_indices,self.cont_indices)] =  sigma_rearranged[np.ix_(self.cont_indices,self.ord_indices)].T
        return sigma_rearranged

    def _init_sigma(self, sigma):
        sigma_new = np.empty(sigma.shape)
        sigma_new[:np.sum(self.ord_indices),:np.sum(self.ord_indices)] = sigma[np.ix_(self.ord_indices,self.ord_indices)]
        sigma_new[np.sum(self.ord_indices):,np.sum(self.ord_indices):] = sigma[np.ix_(self.cont_indices,self.cont_indices)]
        sigma_new[np.sum(self.ord_indices):,:np.sum(self.ord_indices)] = sigma[np.ix_(self.cont_indices,self.ord_indices)] 
        sigma_new[:np.sum(self.ord_indices),np.sum(self.ord_indices):] = sigma[np.ix_(self.ord_indices,self.cont_indices)] 
        self.sigma = sigma_new

    def change_point_test(self, x_batch, decay_coef, nsample=100, max_workers=4):
        n,p = x_batch.shape
        statistics = np.zeros((nsample,3))
        sigma_old = self.get_sigma()
        _, sigma_new = self.partial_fit_and_predict(x_batch, decay_coef=decay_coef, max_workers=max_workers, marginal_update=True, sigma_update=False, sigma_out=True)
        s = self.get_matrix_diff(sigma_old, sigma_new)
        # generate incomplete mixed data samples
        for i in range(nsample):
            np.random.seed(i)
            z = np.random.multivariate_normal(np.zeros(p), sigma_old, n)
            # mask
            x = np.empty(x_batch.shape)
            x[:,self.cont_indices] = self.transform_function.partial_evaluate_cont_observed(z)
            x[:,self.ord_indices] = self.transform_function.partial_evaluate_ord_observed(z)
            loc = np.isnan(x_batch)
            x[loc] = np.nan
            _, sigma = self.partial_fit_and_predict(x, decay_coef=decay_coef, max_workers=max_workers, marginal_update=False, sigma_update=False, sigma_out=True)
            statistics[i,:] = self.get_matrix_diff(sigma_old, sigma)
        # compute test statistics
        pval = np.zeros(3)
        for j in range(3):
            pval[j] = np.sum(s[j]<statistics[:,j])/(nsample+1)
        self._init_sigma(sigma_new)
        return pval, s

        # compute test statistics
    def get_matrix_diff(self, sigma_old, sigma_new, type = 'F'):
        '''
        Return the correlation change tracking statistics, as some matrix norm of normalized matrix difference.
        Support three norms currently: 'F' for Frobenius norm, 'S' for spectral norm and 'N' for nuclear norm. User-defined norm can also be used.
        '''
        p = sigma_old.shape[0]
        u, s, vh = np.linalg.svd(sigma_old)
        factor = (u * np.sqrt(1/s) ) @ vh
        diff = factor @ sigma_new @ factor
        if type == 'F':
            return np.linalg.norm(diff-np.identity(p))
        else:
            _, s, _ = np.linalg.svd(diff)
            if type == 'S':
                return max(abs(s-1))
            if type == 'N':
                return np.sum(abs(s-1))