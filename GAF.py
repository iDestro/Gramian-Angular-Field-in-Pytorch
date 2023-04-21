import torch
class GramianAngularFieldPytorch(object):
    """Gramian Angular Field.

    Parameters
    ----------
    method : 'summation' or 'difference' (default = 'summation')
        Type of Gramian Angular Field. 's' can be used for 'summation'
        and 'd' for 'difference'.

    References
    ----------
    .. [1] Z. Wang and T. Oates, "Encoding Time Series as Images for Visual
           Inspection and Classification Using Tiled Convolutional Neural
           Networks." AAAI Workshop (2015).

    """

    def __init__(self, method='summation'):
        self.method = method

    def min_max_norm(self, X):
        min_val = torch.min(X, dim=-1, keepdim=True)[0]
        max_val = torch.max(X, dim=-1, keepdim=True)[0]
        res = (X - min_val) / (max_val - min_val)
        return res * 2 - 1

    @staticmethod
    def _gasf(X_cos, X_sin):

        X_cos_L = X_cos.unsqueeze(-1)
        X_cos_R = X_cos.unsqueeze(-2)

        X_sin_L = X_sin.unsqueeze(-1)
        X_sin_R = X_sin.unsqueeze(-2)

        X_gasf = torch.bmm(X_cos_L, X_cos_R) - torch.bmm(X_sin_L, X_sin_R)

        return X_gasf

    @staticmethod
    def _gadf(X_cos, X_sin):

        X_sin_L = X_sin.unsqueeze(-1)
        X_cos_R = X_cos.unsqueeze(-2)

        X_cos_L = X_cos.unsqueeze(-1)
        X_sin_R = X_sin.unsqueeze(-2)

        X_gadf = torch.bmm(X_sin_L, X_cos_R) - torch.bmm(X_cos_L, X_sin_R)

        return X_gadf

    def transform(self, X):
        """Transform each time series into a GAF image.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)

        Returns
        -------
        X_new : array-like, shape = (n_samples, image_size, image_size)
            Transformed data. If ``flatten=True``, the shape is
            `(n_samples, image_size * image_size)`.

        """
        X_cos = self.min_max_norm(X)
        X_sin = torch.sqrt(torch.clip(1 - X_cos ** 2, 0, 1))
        if self.method in ['s', 'summation']:
            X_new = self._gasf(X_cos, X_sin)
        else:
            X_new = self._gadf(X_cos, X_sin)
        return X_new