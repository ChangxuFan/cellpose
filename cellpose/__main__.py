"""
Copyright © 2025 Howard Hughes Medical Institute, Authored by Carsen Stringer , Michael Rariden and Marius Pachitariu.
"""
import os, time
import numpy as np
from tqdm import tqdm
from cellpose import utils, models, io, train, transforms
from .version import version_str
from cellpose.cli import get_arg_parser

try:
    from cellpose.gui import gui3d, gui
    GUI_ENABLED = True
except ImportError as err:
    GUI_ERROR = err
    GUI_ENABLED = False
    GUI_IMPORT = True
except Exception as err:
    GUI_ENABLED = False
    GUI_ERROR = err
    GUI_IMPORT = False
    raise

import logging


def main():
    """ Run cellpose from command line
    """

    args = get_arg_parser().parse_args()  # this has to be in a separate file for autodoc to work

    if args.version:
        print(version_str)
        return

    ######## if no image arguments are provided, run GUI or add model and exit ########
    if len(args.dir) == 0 and len(args.image_path) == 0:
        if args.add_model:
            io.add_model(args.add_model)
            return
        else:
            if not GUI_ENABLED:
                print("GUI ERROR: %s" % GUI_ERROR)
                if GUI_IMPORT:
                    print(
                        "GUI FAILED: GUI dependencies may not be installed, to install, run"
                    )
                    print("     pip install 'cellpose[gui]'")
            else:
                if args.Zstack:
                    gui3d.run()
                else:
                    gui.run()
            return

    ############################## run cellpose on images ##############################
    if args.verbose:
        from .io import logger_setup
        logger, log_file = logger_setup()
    else:
        print(
            ">>>> !LOGGING OFF BY DEFAULT! To see cellpose progress, set --verbose")
        print("No --verbose => no progress or info printed")
        logger = logging.getLogger(__name__)


    # find images
    if len(args.img_filter) > 0:
        image_filter = args.img_filter
    else:
        image_filter = None

    device, gpu = models.assign_device(use_torch=True, gpu=args.use_gpu,
                                        device=args.gpu_device)

    if args.pretrained_model is None or args.pretrained_model == "None" or args.pretrained_model == "False" or args.pretrained_model == "0":
        pretrained_model = "cpsam"
        logger.warning("training from scratch is disabled, using 'cpsam' model")
    else:
        pretrained_model = args.pretrained_model

    # Warn users about old arguments from CP3:
    if args.pretrained_model_ortho:
        logger.warning(
            "the '--pretrained_model_ortho' flag is deprecated in v4.0.1+ and no longer used")
    if args.train_size:
        logger.warning("the '--train_size' flag is deprecated in v4.0.1+ and no longer used")
    if args.chan or args.chan2:
        logger.warning('--chan and --chan2 are deprecated, all channels are used by default')
    if args.all_channels:
        logger.warning("the '--all_channels' flag is deprecated in v4.0.1+ and no longer used")
    if args.restore_type:
        logger.warning("the '--restore_type' flag is deprecated in v4.0.1+ and no longer used")
    if args.transformer:
        logger.warning("the '--tranformer' flag is deprecated in v4.0.1+ and no longer used")
    if args.invert:
        logger.warning("the '--invert' flag is deprecated in v4.0.1+ and no longer used")
    if args.chan2_restore:
        logger.warning("the '--chan2_restore' flag is deprecated in v4.0.1+ and no longer used")
    if args.diam_mean:
        logger.warning("the '--diam_mean' flag is deprecated in v4.0.1+ and no longer used")
    if args.train_size:
        logger.warning("the '--train_size' flag is deprecated in v4.0.1+ and no longer used")
    
    if args.norm_percentile is not None:
        value1, value2 = args.norm_percentile
        normalize = {'percentile': (float(value1), float(value2))} 
    else:
        normalize = (not args.no_norm)

    if args.save_each:
        if not args.save_every:
            raise ValueError("ERROR: --save_each requires --save_every")

    if len(args.image_path) > 0 and args.train:
        raise ValueError("ERROR: cannot train model with single image input")

    ## Run evaluation on images
    if not args.train:
        _evaluate_cellposemodel_cli(args, logger, image_filter, device, pretrained_model, normalize)
    
    ## Train a model ##
    else:
        _train_cellposemodel_cli(args, logger, image_filter, device, pretrained_model, normalize)


def _train_cellposemodel_cli(args, logger, image_filter, device, pretrained_model, normalize):
    if args.classifier_mode:
        return _train_classifiermodel_cli(args, logger, image_filter, device, pretrained_model, normalize)

    test_dir = None if len(args.test_dir) == 0 else args.test_dir
    images, labels, image_names, train_probs = None, None, None, None
    test_images, test_labels, image_names_test, test_probs = None, None, None, None
    compute_flows = False
    if len(args.file_list) > 0:
        if os.path.exists(args.file_list):
            dat = np.load(args.file_list, allow_pickle=True).item()
            image_names = dat["train_files"]
            image_names_test = dat.get("test_files", None)
            train_probs = dat.get("train_probs", None)
            test_probs = dat.get("test_probs", None)
            compute_flows = dat.get("compute_flows", False)
            load_files = False
        else:
            logger.critical(f"ERROR: {args.file_list} does not exist")
    else:
        output = io.load_train_test_data(args.dir, test_dir, image_filter,
                                                args.mask_filter,
                                                args.look_one_level_down)
        images, labels, image_names, test_images, test_labels, image_names_test = output
        load_files = True

    # initialize model
    model = models.CellposeModel(device=device, pretrained_model=pretrained_model)

    # train segmentation model
    cpmodel_path = train.train_seg(
            model.net, images, labels, train_files=image_names,
            test_data=test_images, test_labels=test_labels,
            test_files=image_names_test, train_probs=train_probs,
            test_probs=test_probs, compute_flows=compute_flows,
            load_files=load_files, normalize=normalize,
            channel_axis=args.channel_axis, 
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            SGD=args.SGD, n_epochs=args.n_epochs, batch_size=args.train_batch_size,
            min_train_masks=args.min_train_masks,
            nimg_per_epoch=args.nimg_per_epoch,
            nimg_test_per_epoch=args.nimg_test_per_epoch,
            save_path=os.path.realpath(args.dir), 
            save_every=args.save_every,
            save_each=args.save_each,
            model_name=args.model_name_out)[0]
    model.pretrained_model = cpmodel_path
    logger.info(">>>> model trained and saved to %s" % cpmodel_path)
    return model


def _evaluate_cellposemodel_cli(args, logger, imf, device, pretrained_model, normalize):
    # Check with user if they REALLY mean to run without saving anything
    if not args.train:
        saving_something = args.save_png or args.save_tif or args.save_flows or args.save_txt

    if args.mip and (args.do_3D or args.stitch_threshold > 0.0):
        raise ValueError("ERROR: --mip cannot be combined with --do_3D or --stitch_threshold > 0")

    tic = time.time()
    if len(args.dir) > 0:
        image_names = io.get_image_files(
                args.dir, args.mask_filter, imf=imf,
                look_one_level_down=args.look_one_level_down)
    else:
        if os.path.exists(args.image_path):
            image_names = [args.image_path]
        else:
            raise ValueError(f"ERROR: no file found at {args.image_path}")
    nimg = len(image_names)

    if args.savedir:
        if not os.path.exists(args.savedir):
            raise FileExistsError(f"--savedir {args.savedir} does not exist")
        
    logger.info(
            ">>>> running cellpose on %d images using all channels" % nimg)

    if args.classifier_mode:
        return _evaluate_classifiermodel_cli(args, logger, image_names, device, pretrained_model, normalize)

    # handle built-in model exceptions
    model = models.CellposeModel(device=device, pretrained_model=pretrained_model,)

    tqdm_out = utils.TqdmToLogger(logger, level=logging.INFO)

    channel_axis = args.channel_axis
    z_axis = args.z_axis

    def _mip_axis(image, explicit_axis):
        if explicit_axis is not None:
            axis = explicit_axis if explicit_axis >= 0 else image.ndim + explicit_axis
            if axis < 0 or axis >= image.ndim:
                raise ValueError(f"ERROR: --mip_z_axis {explicit_axis} is out of bounds for shape {image.shape}")
            return axis
        # Default projection axis for OME-like stacks.
        return 0

    def _read_2d_image_for_eval(image_name):
        if not args.mip:
            return io.imread_2D(image_name)
        image = io.imread(image_name)
        axis = _mip_axis(image, args.mip_z_axis)
        image = np.max(image, axis=axis)
        logger.info(">>>> applied --mip on axis %d: projected to shape %s", axis, image.shape)
        return transforms.convert_image(image, do_3D=False)

    for image_name in tqdm(image_names, file=tqdm_out):
        eval_channel_axis = channel_axis
        eval_z_axis = z_axis

        if args.do_3D or args.stitch_threshold > 0.:
            logger.info('loading image as 3D zstack')

            # guess at channels/z axis if one is not provided
            if channel_axis or z_axis is None:
                image = io.imread_3D(image_name)
            else:
                image = io.imread(image_name) # Rely on transforms.convert_image() to move channels inside of .eval()
            if channel_axis is None:
                eval_channel_axis = 3
            if z_axis is None:
                eval_z_axis = 0
                
        else:
            image = _read_2d_image_for_eval(image_name)
            eval_channel_axis = None
            eval_z_axis = None
        out = model.eval(
                image, 
                diameter=args.diameter, 
                do_3D=args.do_3D,
                augment=args.augment, 
                flow_threshold=args.flow_threshold,
                cellprob_threshold=args.cellprob_threshold,
                stitch_threshold=args.stitch_threshold, 
                min_size=args.min_size,
                batch_size=args.batch_size,
                bsize=args.bsize,
                resample=not args.no_resample,
                normalize=normalize,
                channel_axis=eval_channel_axis,
                z_axis=eval_z_axis,
                anisotropy=args.anisotropy, 
                niter=args.niter,
                flow3D_smooth=args.flow3D_smooth)
        masks, flows = out[:2]

        if args.exclude_on_edges:
            masks = utils.remove_edge_masks(masks)
        if not args.no_npy:
            io.masks_flows_to_seg(image, masks, flows, image_name,
                                        imgs_restore=None, 
                                        restore_type=None,
                                        ratio=1.)
        if saving_something:
            suffix = "_cp_masks"
            if args.output_name is not None: 
                    # (1) If `savedir` is not defined, then must have a non-zero `suffix`
                if args.savedir is None and len(args.output_name) > 0:
                    suffix = args.output_name
                elif args.savedir is not None and not os.path.samefile(args.savedir, args.dir):
                        # (2) If `savedir` is defined, and different from `dir` then                              
                        # takes the value passed as a param. (which can be empty string)
                    suffix = args.output_name

            io.save_masks(image, masks, flows, image_name,
                                suffix=suffix, png=args.save_png,
                                tif=args.save_tif, save_flows=args.save_flows,
                                save_outlines=args.save_outlines,
                                dir_above=args.dir_above, savedir=args.savedir,
                                save_txt=args.save_txt, in_folders=args.in_folders,
                                save_mpl=args.save_mpl)
        if args.save_rois:
            io.save_rois(masks, image_name)
    logger.info(">>>> completed in %0.3f sec" % (time.time() - tic))

    return model


def _infer_num_classes_from_labels(label_list):
    if label_list is None or len(label_list) == 0:
        raise ValueError("Cannot infer classifier_num_classes without labels")
    max_id = 0
    for lbl in label_list:
        max_id = max(max_id, int(np.asarray(lbl).max()))
    return max_id + 1


def _train_classifiermodel_cli(args, logger, image_filter, device, pretrained_model, normalize):
    from cellpose.nuclei_classifier.model_pixel_classifier import initialize_classifier_model
    from cellpose.nuclei_classifier.train_classifier import train_classifier

    test_dir = None if len(args.test_dir) == 0 else args.test_dir
    if len(args.file_list) > 0:
        raise NotImplementedError("classifier_mode currently does not support --file_list")

    output = io.load_train_test_data(
        args.dir,
        test_dir,
        image_filter,
        args.mask_filter,
        args.look_one_level_down,
    )
    images, labels, image_names, test_images, test_labels, image_names_test = output

    n_classes = args.classifier_num_classes
    if n_classes is None:
        n_classes = _infer_num_classes_from_labels(labels)
        logger.info(">>>> inferred classifier_num_classes=%d from training labels", n_classes)

    model = initialize_classifier_model(
        num_classes=n_classes,
        predict_flows=bool(args.classifier_predict_flows),
        pretrained_model=pretrained_model,
        device=device,
        use_bfloat16=False,
    )

    cpmodel_path = train_classifier(
        model.net,
        train_data=images,
        train_labels=labels,
        train_files=image_names,
        test_data=test_images,
        test_labels=test_labels,
        test_files=image_names_test,
        load_files=True,
        normalize=normalize,
        channel_axis=args.channel_axis,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        n_epochs=args.n_epochs,
        batch_size=args.train_batch_size,
        nimg_per_epoch=args.nimg_per_epoch,
        nimg_test_per_epoch=args.nimg_test_per_epoch,
        save_path=os.path.realpath(args.dir),
        save_every=args.save_every,
        save_each=args.save_each,
        model_name=args.model_name_out,
        compute_flows=bool(args.classifier_compute_flows),
        flow_loss_weight=float(args.classifier_flow_weight),
    )[0]

    model.pretrained_model = cpmodel_path
    logger.info(">>>> classifier model trained and saved to %s", cpmodel_path)
    return model


def _evaluate_classifiermodel_cli(args, logger, image_names, device, pretrained_model, normalize):
    from cellpose.nuclei_classifier.model_pixel_classifier import initialize_classifier_model

    n_classes = args.classifier_num_classes
    if n_classes is None:
        raise ValueError("--classifier_num_classes is required for classifier inference")

    model = initialize_classifier_model(
        num_classes=n_classes,
        predict_flows=bool(args.classifier_predict_flows),
        pretrained_model=pretrained_model,
        device=device,
        use_bfloat16=False,
    )

    if args.mip and args.do_3D:
        raise ValueError("ERROR: --mip cannot be combined with --do_3D for classifier inference")

    tic = time.time()
    suffix = args.classifier_output_suffix if args.classifier_output_suffix is not None else "_cls"
    out_dir = args.savedir if args.savedir else None

    def _mip_axis(image, explicit_axis):
        if explicit_axis is not None:
            axis = explicit_axis if explicit_axis >= 0 else image.ndim + explicit_axis
            if axis < 0 or axis >= image.ndim:
                raise ValueError(f"ERROR: --mip_z_axis {explicit_axis} is out of bounds for shape {image.shape}")
            return axis
        return 0

    def _read_classifier_image(image_name):
        if not args.mip:
            return io.imread_2D(image_name), args.channel_axis, args.z_axis

        image = io.imread(image_name)
        axis = _mip_axis(image, args.mip_z_axis)
        image = np.max(image, axis=axis)
        logger.info(">>>> applied --mip on axis %d: projected to shape %s", axis, image.shape)
        image = transforms.convert_image(image, do_3D=False)
        return image, None, None

    def _custom_classifier_output_name(image_name):
        if args.classifier_output_name is None:
            stem, _ = os.path.splitext(os.path.basename(image_name))
            return f"{stem}{suffix}.tif"

        name = args.classifier_output_name
        root, ext = os.path.splitext(name)
        if ext == "":
            ext = ".tif"

        if len(image_names) == 1:
            return f"{root}{ext}"

        stem, _ = os.path.splitext(os.path.basename(image_name))
        return f"{root}_{stem}{ext}"

    tqdm_out = utils.TqdmToLogger(logger, level=logging.INFO)
    for image_name in tqdm(image_names, file=tqdm_out):
        image, eval_channel_axis, eval_z_axis = _read_classifier_image(image_name)
        class_map, _, _, _ = model.eval(
            image,
            batch_size=args.batch_size,
            channel_axis=eval_channel_axis,
            z_axis=eval_z_axis,
            normalize=normalize,
            invert=args.invert,
            rescale=None,
            tile_overlap=0.1,
            bsize=args.bsize,
        )

        if args.save_tif:
            out_name = _custom_classifier_output_name(image_name)
            out_path = os.path.join(out_dir, out_name) if out_dir else os.path.join(os.path.dirname(image_name), out_name)
            io.imsave(out_path, class_map.astype(np.uint16))

    logger.info(">>>> classifier inference completed in %0.3f sec", (time.time() - tic))
    return model


if __name__ == "__main__":
    main()
