import PIL
from ibug.face_detection import RetinaFacePredictor
from ibug.face_parsing import FaceParser as RTNetPredictor
import dlib
import numpy as np
import cv2
import os
import random
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import torch
from torch import nn, autograd
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import models, transforms

from utils import common, train_utils
from criteria import id_loss, moco_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from criteria.lpips.lpips import LPIPS
from models.psp import pSp
from models.latent_codes_pool import LatentCodesPool
from models.discriminator import LatentCodesDiscriminator
from models.encoders.psp_encoders import ProgressiveStage
from training.ranger import Ranger

random.seed(0)
torch.manual_seed(0)


class Coach:
    def __init__(self, opts, prev_train_checkpoint=None):
        self.softmax = torch.nn.Softmax()
        self.opts = opts
        self.opts.classifier_lambda = 0.2 

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device
        # Initialize network
        self.net = pSp(self.opts).to(self.device)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.face_detector = RetinaFacePredictor(threshold=0.8, device="cuda",
                                            model=(RetinaFacePredictor.get_model('mobilenet0.25')))
        self.face_parser = RTNetPredictor(
            device="cuda", ckpt=None, encoder="rtnet50", decoder="fcn", num_classes=11)

        # Initialize loss
        if self.opts.lpips_lambda > 0:
            self.lpips_loss = LPIPS(net_type=self.opts.lpips_type).to(self.device).eval()
        if self.opts.id_lambda > 0:
            #if 'ffhq' in self.opts.dataset_type or 'celeb' in self.opts.dataset_type:
            self.id_loss = id_loss.IDLoss().to(self.device).eval()
            #else:
            #    self.id_loss = moco_loss.MocoLoss(opts).to(self.device).eval()
        self.mse_loss = nn.MSELoss().to(self.device).eval()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()

        # Initialize discriminator
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator = LatentCodesDiscriminator(512, 4).to(self.device)
            self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()),
                                                            lr=opts.w_discriminator_lr)
            self.real_w_pool = LatentCodesPool(self.opts.w_pool_size)
            self.fake_w_pool = LatentCodesPool(self.opts.w_pool_size)

        # Initialize dataset
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

        if prev_train_checkpoint is not None:
            self.load_from_train_checkpoint(prev_train_checkpoint)
            prev_train_checkpoint = None

        model_ft, input_size = self.initialize_model("resnet", 2, True, use_pretrained=True)
        self.model_ft = model_ft
        self.input_size = input_size
        data_dict = torch.load("/home/yakovdan/best_classifier.pt")
        self.model_ft.load_state_dict(data_dict['state_dict'])
        self.model_ft = self.model_ft.cuda()
        self.model_ft.eval()
        self.classifier_transforms = transforms.Compose([transforms.Resize(self.input_size),
                                                transforms.CenterCrop(self.input_size),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.classifier_transforms1 = transforms.Compose([transforms.Resize(self.input_size),
                                                         transforms.CenterCrop(self.input_size),
                                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                                              [0.229, 0.224, 0.225])])
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet18
            """
            model_ft = models.resnet18(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    def classifier_loss(self, image):
        transformed_image = self.classifier_transforms1(image)
        logits_vec = self.model_ft(transformed_image)
        s_max = self.softmax(logits_vec)
        return torch.pow((1 - s_max[0][1]), 3)

    def load_from_train_checkpoint(self, ckpt):
        print('Loading previous training data...')
        self.global_step = 0#ckpt['global_step'] + 1
        self.best_val_loss = 100#ckpt['best_val_loss']
        self.net.load_state_dict(ckpt['state_dict'])

        if self.opts.keep_optimizer:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.opts.w_discriminator_lambda > 0:
            self.discriminator.load_state_dict(torch.load('/home/yakovdan/encoder4editing/pretrained_models/ffhq_discriminator_ckpt.pt'))
            #self.discriminator_optimizer.load_state_dict(ckpt['discriminator_optimizer_state_dict'])
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update(is_resume_from_ckpt=True)
        print(f'Resuming training from step {self.global_step}')

    def train(self):
        self.net.train()
        if self.opts.progressive_steps:
            self.check_for_progressive_training_update()
        while self.global_step < self.opts.max_steps:
            for batch_idx, batch in enumerate(self.train_dataloader):
                loss_dict = {}
                if self.is_training_discriminator():
                    loss_dict = self.train_discriminator(batch)
                x, y, y_hat, latent = self.forward(batch)
                loss, encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                loss_dict = {**loss_dict, **encoder_loss_dict}
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (
                        self.global_step < 1000 and self.global_step % 25 == 0):
                    self.parse_and_log_images(id_logs, x, y, y_hat, title='images/train/faces')
                if self.global_step % self.opts.board_interval == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps:
                    print('OMG, finished training!')
                    break

                self.global_step += 1
                if self.opts.progressive_steps:
                    self.check_for_progressive_training_update()

    def check_for_progressive_training_update(self, is_resume_from_ckpt=False):
        for i in range(len(self.opts.progressive_steps)):
            if is_resume_from_ckpt and self.global_step >= self.opts.progressive_steps[i]:  # Case checkpoint
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))
            if self.global_step == self.opts.progressive_steps[i]:   # Case training reached progressive step
                self.net.encoder.set_progressive_stage(ProgressiveStage(i))

    def validate(self):
        self.net.eval()
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            cur_loss_dict = {}
            if self.is_training_discriminator():
                cur_loss_dict = self.validate_discriminator(batch)
            with torch.no_grad():
                x, y, y_hat, latent = self.forward(batch)
                loss, cur_encoder_loss_dict, id_logs = self.calc_loss(x, y, y_hat, latent)
                cur_loss_dict = {**cur_loss_dict, **cur_encoder_loss_dict}
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(id_logs, x, y, y_hat,
                                      title='images/test/faces',
                                      subscript='{:04d}'.format(batch_idx))

            # For first step just do sanity test on small amount of data
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train()
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train()
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.encoder.parameters())
        if self.opts.train_decoder:
            params += list(self.net.decoder.parameters())
        else:
            self.requires_grad(self.net.decoder, False)
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else:
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
                                      target_root=dataset_args['train_target_root'],
                                      source_transform=transforms_dict['transform_source'],
                                      target_transform=transforms_dict['transform_gt_train'],
                                      opts=self.opts)
        test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
                                     target_root=dataset_args['test_target_root'],
                                     source_transform=transforms_dict['transform_source'],
                                     target_transform=transforms_dict['transform_test'],
                                     opts=self.opts)
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent):
        loss_dict = {}
        loss = 0.0
        id_logs = None
        if self.is_training_discriminator():  # Adversarial loss
            loss_disc = 0.
            dims_to_discriminate = self.get_dims_to_discriminate() if self.is_progressive_training() else \
                list(range(self.net.decoder.n_latent))

            for i in dims_to_discriminate:
                w = latent[:, i, :]
                fake_pred = self.discriminator(w)
                loss_disc += F.softplus(-fake_pred).mean()
            loss_disc /= len(dims_to_discriminate)
            loss_dict['encoder_discriminator_loss'] = float(loss_disc)
            loss += self.opts.w_discriminator_lambda * loss_disc

        if self.opts.progressive_steps and self.net.encoder.progressive_stage.value != 18:  # delta regularization loss
            total_delta_loss = 0
            deltas_latent_dims = self.net.encoder.get_deltas_starting_dimensions()

            first_w = latent[:, 0, :]
            for i in range(1, self.net.encoder.progressive_stage.value + 1):
                curr_dim = deltas_latent_dims[i]
                delta = latent[:, curr_dim, :] - first_w
                delta_loss = torch.norm(delta, self.opts.delta_norm, dim=1).mean()
                loss_dict[f"delta{i}_loss"] = float(delta_loss)
                total_delta_loss += delta_loss
            loss_dict['total_delta_loss'] = float(total_delta_loss)
            loss += self.opts.delta_norm_lambda * total_delta_loss

        if self.opts.id_lambda > 0:  # Similarity loss # MODIFY THIS
            loss_id, sim_improvement, id_logs = self.id_loss(self.mask_eyes(y_hat), self.mask_eyes(y), self.mask_eyes(x))
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0: # THIS
            loss_l2 = F.mse_loss(self.mask_eyes(y_hat), self.mask_eyes(y))
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda * 2
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat, y)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.classifier_lambda > 0:
            #print("Y_hat type: "+str(type(y_hat)))
            classifier_loss = self.classifier_loss(y_hat)
            loss_dict['loss_classifier'] = float(classifier_loss)
            loss += classifier_loss * self.opts.classifier_lambda
        loss_dict['loss'] = float(loss)
        return loss, loss_dict, id_logs

    def forward(self, batch):
        x, y = batch
        x, y = x.to(self.device).float(), y.to(self.device).float()
        y_hat, latent = self.net.forward(x, return_latents=True)
        if self.opts.dataset_type == "cars_encode":
            y_hat = y_hat[:, :, 32:224, :]
        return x, y, y_hat, latent

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
        im_data = []
        for i in range(display_count):
            cur_im_data = {
                'input_face': common.log_input_image(x[i], self.opts),
                'target_face': common.tensor2im(y[i]),
                'output_face': common.tensor2im(y_hat[i]),
            }
            if id_logs is not None:
                for key in id_logs[i]:
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.net.state_dict(),
            'opts': vars(self.opts)
        }
        # save the latent avg in state_dict for inference if truncation of w was used during training
        if self.opts.start_from_latent_avg:
            save_dict['latent_avg'] = self.net.latent_avg

        if self.opts.save_training_data:  # Save necessary information to enable training continuation from checkpoint
            save_dict['global_step'] = self.global_step
            save_dict['optimizer'] = self.optimizer.state_dict()
            save_dict['best_val_loss'] = self.best_val_loss
            if self.opts.w_discriminator_lambda > 0:
                save_dict['discriminator_state_dict'] = self.discriminator.state_dict()
                save_dict['discriminator_optimizer_state_dict'] = self.discriminator_optimizer.state_dict()
        return save_dict

    def get_dims_to_discriminate(self):
        deltas_starting_dimensions = self.net.encoder.get_deltas_starting_dimensions()
        return deltas_starting_dimensions[:self.net.encoder.progressive_stage.value + 1]

    def is_progressive_training(self):
        return self.opts.progressive_steps is not None

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Discriminator ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

    def is_training_discriminator(self):
        return self.opts.w_discriminator_lambda > 0

    @staticmethod
    def discriminator_loss(real_pred, fake_pred, loss_dict):
        real_loss = F.softplus(-real_pred).mean()
        fake_loss = F.softplus(fake_pred).mean()

        loss_dict['d_real_loss'] = float(real_loss)
        loss_dict['d_fake_loss'] = float(fake_loss)

        return real_loss + fake_loss

    @staticmethod
    def discriminator_r1_loss(real_pred, real_w):
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_w, create_graph=True
        )
        grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

        return grad_penalty

    @staticmethod
    def requires_grad(model, flag=True):
        for p in model.parameters():
            p.requires_grad = flag

    def train_discriminator(self, batch):
        loss_dict = {}
        x, _ = batch
        x = x.to(self.device).float()
        self.requires_grad(self.discriminator, True)

        with torch.no_grad():
            real_w, fake_w = self.sample_real_and_fake_latents(x)
        real_pred = self.discriminator(real_w)
        fake_pred = self.discriminator(fake_w)
        loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
        loss_dict['discriminator_loss'] = float(loss)

        self.discriminator_optimizer.zero_grad()
        loss.backward()
        self.discriminator_optimizer.step()

        # r1 regularization
        d_regularize = self.global_step % self.opts.d_reg_every == 0
        if d_regularize:
            real_w = real_w.detach()
            real_w.requires_grad = True
            real_pred = self.discriminator(real_w)
            r1_loss = self.discriminator_r1_loss(real_pred, real_w)

            self.discriminator.zero_grad()
            r1_final_loss = self.opts.r1 / 2 * r1_loss * self.opts.d_reg_every + 0 * real_pred[0]
            r1_final_loss.backward()
            self.discriminator_optimizer.step()
            loss_dict['discriminator_r1_loss'] = float(r1_final_loss)

        # Reset to previous state
        self.requires_grad(self.discriminator, False)

        return loss_dict

    def validate_discriminator(self, test_batch):
        with torch.no_grad():
            loss_dict = {}
            x, _ = test_batch
            x = x.to(self.device).float()
            real_w, fake_w = self.sample_real_and_fake_latents(x)
            real_pred = self.discriminator(real_w)
            fake_pred = self.discriminator(fake_w)
            loss = self.discriminator_loss(real_pred, fake_pred, loss_dict)
            loss_dict['discriminator_loss'] = float(loss)
            return loss_dict

    def sample_real_and_fake_latents(self, x):
        sample_z = torch.randn(self.opts.batch_size, 512, device=self.device)
        real_w = self.net.decoder.get_latent(sample_z)
        fake_w = self.net.encoder(x)
        if self.opts.start_from_latent_avg:
            fake_w = fake_w + self.net.latent_avg.repeat(fake_w.shape[0], 1, 1)
        if self.is_progressive_training():  # When progressive training, feed only unique w's
            dims_to_discriminate = self.get_dims_to_discriminate()
            fake_w = fake_w[:, dims_to_discriminate, :]
        if self.opts.use_w_pool:
            real_w = self.real_w_pool.query(real_w)
            fake_w = self.fake_w_pool.query(fake_w)
        if fake_w.ndim == 3:
            fake_w = fake_w[:, 0, :]
        return real_w, fake_w

    def face_area(self, face):
        return (face[2] - face[0]) * (face[3] - face[1])

    def predicate(self, faces):
        l = faces.shape[0]
        areas = np.zeros(l, dtype=np.float32)
        for i in range(faces.shape[0]):
            a = self.face_area(faces[i])
            areas[i] = a
        return areas

    def segment_face(self, frame):
        frame = cv2.resize(frame, dsize=(256, 256), interpolation=cv2.INTER_LANCZOS4)
        faces = self.face_detector(frame, rgb=True)
        if len(faces) > 1:
            items = self.predicate(faces)
            order = np.argsort(items)
            order = np.flip(order, axis=0)
            faces = faces[order]
            faces = faces[0].reshape((1, 15))

        if len(faces) == 0:
            print('no faces found, not masking anything')
            #np.save('frame_with_issue', frame)           
            frame_copy = np.ones_like(frame)
            return frame_copy


        # Parse faces
        masks = self.face_parser.predict_img(frame, faces, rgb=True).astype(np.uint8)
        frame_copy = np.ones_like(frame)
        i1 = (masks[0, :, :] == 4)
        i2 = (masks[0, :, :] == 5)
        ind = (i1 | i2)  # left or right eye
        #ind2 = (masks[0, :, :] != 4 & masks[0, :, :] != 5)  # not any of the eyes
        frame_copy[:, :, 0][ind] = 0
        frame_copy[:, :, 1][ind] = 0
        frame_copy[:, :, 2][ind] = 0
        # frame_copy[:, :, 0][ind2] = 255
        # frame_copy[:, :, 1][ind2] = 255
        # frame_copy[:, :, 2][ind2] = 255

        # frame_thresh = 255 * np.ones((256, 256), dtype=np.uint8)
        # frame_thresh[ind] = 0
        #
        # thresh = cv2.threshold(frame_thresh, 0, 255,
        #                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # numLabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
        # cc_areas = [stats[x, cv2.CC_STAT_AREA] for x in range(1, numLabels)]
        # largest_label_by_area = np.argmax(np.array(cc_areas)) + 1
        # ind3 = (labels == largest_label_by_area)
        # ind4 = (labels != largest_label_by_area)
        # frame_copy = np.copy(frame)
        # frame_copy[:, :, 0][ind] = 0
        # frame_copy[:, :, 1][ind] = 0
        # frame_copy[:, :, 2][ind] = 0
        # frame_copy[:, :, 0][ind4] = 0
        # frame_copy[:, :, 1][ind4] = 0
        # frame_copy[:, :, 2][ind4] = 0
        return frame_copy



    def mask_eyes(self, input_tensor):
        # input tensor is (b, c, h, w)
        #print("masking eyes")
        #torch.save(input_tensor, 'input_tensor.pt')
        torch_mask = torch.ones_like(input_tensor)
        for i in range(input_tensor.shape[0]):
            image_tensor = input_tensor[i]
            input_np_array = image_tensor.cpu().detach().numpy()
            input_np_array = (input_np_array + 1) * 127.5
            input_np_array = input_np_array.astype(np.uint8)
            input_np_array = input_np_array.transpose([1, 2, 0])
            #np.save(f'input_np_array_{i}', input_np_array)
            mask = self.segment_face(input_np_array).transpose([2, 0, 1])
            torch_mask[i] *= torch.from_numpy(mask).cuda()
        #torch.save(torch_mask, "/home/yakovdan/win/exp_16/torch_mask.pt")
        masked_tensor = input_tensor * torch_mask
        return masked_tensor
