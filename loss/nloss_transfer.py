from loss import nadv_loss, CORAL, ncos, nkl_js, nmmd, nmutual_info, cosine, pairwise_dist

class TransferLoss(object):
    def __init__(self, loss_type='ncosine', input_dim=512):
        self.loss_type = loss_type
        self.input_dim = input_dim

    def compute(self, X, Y):

        if self.loss_type =='mmd_lin' or self.loss_type == 'mmd':
            mmdloss = nmmd.MMD_loss(kernel_type='linear')
            loss = mmdloss(X, Y)
        elif self.loss_type == 'coral':
            loss = CORAL(X, Y)
        elif self.loss_type == 'cosine' or self.loss_type == 'cos':
            loss = 1 - cosine(X, Y)
        elif self.loss_type == 'kl':
            loss = nkl_js.kl_div(X, Y)
        elif self.loss_type == 'js':
            loss = nkl_js.js(X, Y)
        elif self.loss_type == 'mine':
            mine_model = nmutual_info.Mine_estimator(
                input_dim=self.input_dim, hidden_dim=60).to(device)
            loss = mine_model(X, Y)
        elif self.loss_type == 'adv':
            loss = nadv_loss.adv(X, Y, input_dim=self.input_dim, hidden_dim=32)
        elif self.loss_type == 'mmd_rbf':
            mmdloss = nmmd.MMD_loss(kernel_type='rbf')
            loss = mmdloss(X, Y)
        elif self.loss_type == 'pairwise':
            pair_mat = pairwise_dist(X, Y)
            import torch
            loss = torch.norm(pair_mat)

        return loss
if __name__ == "__main__":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trans_loss = TransferLoss('adv')
    a = (torch.randn(5, 512) * 10).to(device)
    b = (torch.randn(5, 512) * 10).to(device)
    print(trans_loss.compute(a, b))
