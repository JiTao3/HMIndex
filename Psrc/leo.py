import torch
import torch.nn as nn

from model import LeoModel
from train_dataset import loadAllSplitIndexTask, get_batch

# from torchmeta.utils.data import BatchMetaDataLoader

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")


class LEO:
    def __init__(
        self,
        N,
        K,
        batch_size,
        embedding_size,
        hidden_size,
        drop_out,
        inner_lr,
        fintune_lr,
        cuda,
        out_lr,
        kl_weight,
        encoder_penalty_weight,
        orthogonality_penalty_weight,
        l2_penalty_weight,
        inner_update_epoch,
        out_update_epoch,
        total_step,
        clip_vale,
    ) -> None:
        self.data_set = None
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.drop_out = drop_out
        self.inner_lr = inner_lr
        self.fintune_lr = fintune_lr
        self._cuda = cuda
        self.model = LeoModel(
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            drop_out=self.drop_out,
            inner_lr=self.inner_lr,
            fintune_lr=self.fintune_lr,
            cuda=self._cuda,
        )

        if self._cuda:
            self.model.to(device)
        self.N = N
        self.K = K
        self.batch_size = batch_size
        self.out_lr = out_lr
        self.kl_weight = kl_weight
        self.encoder_penalty_weight = encoder_penalty_weight
        self.orthogonality_penalty_weight = orthogonality_penalty_weight
        self.l2_penalty_weight = l2_penalty_weight
        self.inner_update_epoch = inner_update_epoch
        self.out_update_epoch = out_update_epoch
        self.total_step = total_step
        self.clip_vale = clip_vale

        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def run_batch(self, batch, step, train=True):
        # inner_training (inner loop)
        latents, kl_div, encoder_penalty = self.meta_train_batch(
            batch["train"]["input"], batch["train"]["target"]
        )
        # inner_fintune & task-validate
        # todo xxx
        val_loss = self.finetune_validation(
            latents,
            batch["train"]["input"],
            batch["train"]["target"],
            batch["val"]["input"],
            batch["val"]["target"],
        )
        orthogonality_penalty = self.orthogonality(
            list(self.model.decoder.parameters())[0]
        )

        total_loss = (
            val_loss
            + self.kl_weight * kl_div
            + self.encoder_penalty_weight * encoder_penalty
            + self.orthogonality_penalty_weight * orthogonality_penalty
        )
        return (
            total_loss,
            val_loss,
            kl_div,
            encoder_penalty,
            orthogonality_penalty,
        )

    def meta_train_batch(self, inputs, target):
        latents, kl_div = self.model.encode(inputs)
        latents_init = latents

        for i in range(self.inner_update_epoch):
            latents.retain_grad()
            regression_weight = self.model.decode(latents)
            train_loss = self.model.cal_target_loss(
                inputs, regression_weight, target
            )
            train_loss.backward(retain_graph=True)

            latents = latents - self.model.inner_l_rate * latents.grad.data

        encode_penalty = torch.mean((latents_init - latents) ** 2)
        return latents, kl_div, encode_penalty

    def finetune_validation(
        self, latents, inputs, target, val_input, val_target
    ):
        regression_weight = self.model.decode(latents)
        regression_weight.retain_grad()
        train_loss = self.model.cal_target_loss(
            inputs, regression_weight, target
        )
        for j in range(self.out_update_epoch):
            train_loss.backward(retain_graph=True)
            regression_weight = (
                regression_weight
                - self.model.fintune_lr * regression_weight.grad
            )
            regression_weight.retain_grad()
            train_loss = self.model.cal_target_loss(
                inputs, regression_weight, target
            )
            print(
                "Step{}, Training Loss:{:.6f}, Inner LR:{}, FIntune LR:{}".format(
                    j,
                    train_loss.item(),
                    float(self.model.inner_l_rate),
                    float(self.model.fintune_lr),
                )
            )
        val_loss = self.model.cal_target_loss(
            val_input, regression_weight, val_target
        )
        return val_loss

    def orthogonality(self, weight):
        w2 = torch.mm(weight, weight.transpose(0, 1))
        wn = torch.norm(weight, dim=1, keepdim=True) + 1e-20
        correlation_matrix = w2 / torch.mm(wn, wn.transpose(0, 1))
        assert correlation_matrix.size(0) == correlation_matrix.size(1)
        I = torch.eye(correlation_matrix.size(0)).cuda(device=device)
        return torch.mean((correlation_matrix - I) ** 2)

    def train(self, task_path):
        # load traing data
        # load split data
        tasks = loadAllSplitIndexTask(task_path)

        # tasksets = IndexDataset(tasks, max_size_task)
        # dataloader = BatchMetaDataLoader(tasksets, batch_size=task_per_batch, pin_memory=True)
        lr_list = ["inner_l_rate", "fintune_lr"]
        params = [
            x[1]
            for x in list(
                filter(
                    lambda kv: kv[0] not in lr_list,
                    self.model.named_parameters(),
                )
            )
        ]
        lr_params = [
            x[1]
            for x in list(
                filter(
                    lambda kv: kv[0] in lr_list, self.model.named_parameters()
                )
            )
        ]
        optim = torch.optim.Adam(
            params,
            lr=self.out_lr,
            weight_decay=self.l2_penalty_weight,
        )
        optim_lr = torch.optim.Adam(lr_params, lr=self.out_lr)
        for step in range(self.total_step):
            optim.zero_grad()
            optim_lr.zero_grad()
            batch = get_batch(
                tasks, self.K, self.batch_size, self.N, device=device
            )
            (
                total_loss,
                val_loss,
                kl_div,
                encoder_penalty,
                orthogonality_penalty,
            ) = self.run_batch(batch, step)
            print(
                "(Meta-Valid) [Step: %d/%d] Total Loss: %4.4f Valid Loss: %4.4f"
                % (step, self.total_step, total_loss.item(), val_loss.item())
            )
            total_loss.backward()

            nn.utils.clip_grad_value_(self.model.parameters(), self.clip_vale)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_vale)
            optim.step()
            optim_lr.step()
            if float(self.model.inner_l_rate) > self.inner_lr * 1.2:
                self.model.inner_l_rate.data = torch.FloatTensor([self.inner_lr])
            if float(self.model.fintune_lr) > self.fintune_lr * 10:
                self.model.fintune_lr.data = torch.FloatTensor([self.fintune_lr])
            self.model.to(device)
    def test(self):
        pass
        # TODO

    def save(self, save_path):
        torch.save(self.model, save_path)


if __name__ == "__main__":
    model = torch.load("model/simple_train_1k_1.pt", map_location=device)
    print(1)
