import os

import onnx
import onnxoptimizer
import torch
from torch.autograd import Variable

import util.util as util
from data import create_dataset
from models import create_model
from options.test_options import TestOptions

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    # create a webpage for viewing the results

    for i, data in enumerate(dataset):
        if i == 0:
            model.data_dependent_initialize(data)
            model.setup(opt)               # regular setup: load and print networks; create schedulers
            # model.parallelize()
            if opt.eval:
                model.eval()
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        # store image
        for label, im_data in visuals.items():
            im = util.tensor2im(im_data)
            image_name = '%s/%s.png' % (label, os.path.splitext(os.path.basename((img_path[0])))[0])
            os.makedirs(os.path.join("results", label), exist_ok=True)
            save_path = os.path.join("results", image_name)
            util.save_image(im, save_path, aspect_ratio=1.0)

    # convert model
    print("exporting...")
    dummy_input = Variable(torch.randn(1, 3, 512, 512)).cuda()
    torch.onnx.export(model.netG, dummy_input, "human-back-texture-CUT-512.onnx")
    print("done!")

    exit()

    print("optimize")
    src_onnx = 'human-back-texture-CUT-512.onnx'
    opt_onnx = 'human-back-texture-CUT-512.opt.onnx'

    # load model
    model = onnx.load(src_onnx)

    # optimize
    onnx.checker.check_model(model)
    # inferred_model = onnx.shape_inference.infer_shapes(model)
    # model = onnx.optimizer.optimize(inferred_model, passes=['fuse_bn_into_conv'])

    for init in model.graph.initializer:
        for value_info in model.graph.value_info:
            if init.name == value_info.name:
                model.graph.input.append(value_info)

    model = onnxoptimizer.optimize(model, ['fuse_bn_into_conv'])

    # save optimized model
    with open(opt_onnx, "wb") as f:
        f.write(model.SerializeToString())
    print("done")
