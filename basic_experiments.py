##################################################################################################################
## I built on top of the examples file provided in the innvestigate lib and added my own experiments
##################################################################################################################


import argparse
import os
import time
import ipdb

import warnings
warnings.simplefilter('ignore')

from importlib import util
import numpy as np
from scipy import ndimage
from skimage.transform import warp, AffineTransform

import keras
import keras.backend
import keras.models

import innvestigate
import innvestigate.applications.imagenet
import innvestigate.utils as iutils

import my_utils as eutils

import my_utils_imagenet as imgnetutils

def get_arguments():
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Input paramters for image explanation')

    # Add the paramters positional/optional (here only optional)
    parser.add_argument( '-ip', '--img_path', help='Path to the image (should be **/wnid/****.jpg/jpeg). Otherwise original image label is considered to be 1000/others ' )
    parser.add_argument( '-n', '--netname', 
                         help='Name of the net you want to analyze (options - vgg16, vgg19, resnet50, inception_v3, densenet169, nasnet_large)', 
                         default="vgg16" )
    
    # "vgg16_custom",
    # "vgg16",
    # "vgg19",
    # "resnet50",
    # "inception_v3",
    # "inception_resnet_v2",
    # "densenet121",
    # "densenet169",
    # "densenet201",
    # "nasnet_large",
    # "nasnet_mobile"
    
    
    parser.add_argument( '-ier', '--img_edit_row', help='Hide rows (one at a time) of ROI/use True/t/T/y/Y/1', nargs='+' )
    parser.add_argument( '-iec', '--img_edit_col', help='Hide coloumns (one at a time) of ROI/use True/t/T/y/Y/1', nargs='+' )
    parser.add_argument( '-vf', '--vertical_flip', help='Flip the image vertically' )
    parser.add_argument( '-hf', '--horizontal_flip', help='Flip the image horizontally' )
    parser.add_argument( '-ra', '--rotation_angle', help='Rotate the image by given angle' )
    parser.add_argument( '-t', '--translate_image', help='Translate the image. Provide translation in both x and y directions', nargs='+', type=int )
    
    parser.add_argument( '-l', '--labels', type=str, 
                         help='Select the label for which you want to find the importance (options - Ture, pred, topN, bottomN random_num[0-999] )', 
                         nargs='+',
                         default="True" )
    
    parser.add_argument( '-o', '--out_path', help='Path to the of the output image', default='./' )
    parser.add_argument( '-s', '--save', help='Whether to save the results or not', default='True' )
    


    # Parse the arguments
    args = parser.parse_args()
    
    args.out_path = os.path.abspath(args.out_path)
    if args.out_path[-1] != '/':
        args.out_path += '/'
      
    
    return args

def main(args):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(base_dir)
    
    # Load the model definition.
    
    tmp = getattr(innvestigate.applications.imagenet, os.environ.get("NETWORKNAME", args.netname) )
    net = tmp(load_weights=True, load_patterns="relu")

    # Build the model.
    model = keras.models.Model(inputs=net["in"], outputs=net["sm_out"])
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    # Handle input depending on model and backend.
    channels_first = keras.backend.image_data_format() == "channels_first"
    color_conversion = "BGRtoRGB" if net["color_coding"] == "BGR" else None
	
    print('model loaded')
    
    ## get class_name to label mappings
    with open("imagenet_label_mapping") as f:
        label_to_class_name = {int(x.split(":")[0]): x.split(":")[1].strip()
                               for x in f.readlines() if len(x.strip()) > 0}
                               
    #label_to_class_name = eutils.get_imagenet_data(net["image_shape"][0])[1] # This to store only the last output
    label_to_class_name[1000] = 'Others'
    
    
    if args.img_path is None:
        raise Exception("Image path not given. Please provide a path")
   
    else:
        # Get some example test set images.
        args.img_path = os.path.abspath(args.img_path)
        
        print('Image from given path')
        synsetID = args.img_path.split("/")[-2]
        imgName = args.img_path.split("/")[-1]
        size = net["image_shape"][0]
        print(size)
            
        # to get class number from synsetID
        with open("synset_id_to_class") as f:
            synset_to_class = {x.split()[1]: int(x.split()[0])
                               for x in f.readlines() if len(x.strip()) > 0}

        def get_class(f):
            # File from ImageNet training sets
            ret = synset_to_class.get(f.split("_")[0], None)
            if ret is None:
                # Random JPEG file
                ret = 1000
            return ret
        
        # It's just a single image (not images)
        images = [ ( eutils.load_image(args.img_path, size), get_class(imgName) ) ]
        
        
        out_img_name = ['no_edit']
    
    ## Looking back, this section could have been implemented in a much better way using functions
    if args.img_edit_row is not None and args.img_edit_col is not None:
        print('In both edit')
        img , class_label = images[0]
        (r, c, z) = np.shape(img)
        
        ## Editting rows
        if len(args.img_edit_row) == 1:
            if args.img_edit_row[0].isdigit():
                sIdx = int(args.img_edit_row[0])
                eIdx = int(args.img_edit_row[0]) + 1
            else:
                sIdx = 0
                eIdx = r
        elif len(args.img_edit_row) == 2:
            sIdx = int(args.img_edit_row[0])
            eIdx = int(args.img_edit_row[1])+1
        else:
            raise Exception("Specify a range ")
            
        for i in range(sIdx, eIdx):
            temp_img = np.copy(img)
            temp_img[i, :, :] = 0*temp_img[i, :, :]
            images.append((temp_img, class_label))
            out_img_name.append('row_edit'+ str(i))
            
        ## Editting coloumns 
        if len(args.img_edit_col) == 1:
            if args.img_edit_col[0].isdigit():
                sIdx = int(args.img_edit_col[0])
                eIdx = int(args.img_edit_col[0]) + 1
            else:
                sIdx = 0
                eIdx = c
        elif len(args.img_edit_col) == 2:
            sIdx = int(args.img_edit_col[0])
            eIdx = int(args.img_edit_col[1])+1
        else:
            raise Exception("Specify a range ")
            
        for i in range(sIdx, eIdx):
            temp_img = np.copy(img)
            temp_img[:, i, :] = 0*temp_img[:, i, :]
            images.append((temp_img, class_label))
            out_img_name.append('col_edit' + str(i))
        
    # Row wise editing of the image
    elif args.img_edit_row is not None:
        print('In row edit')
        img , class_label = images[0]
        (r, c, z) = np.shape(img)
        
        ## Editting rows
        if len(args.img_edit_row) == 1:
            if args.img_edit_row[0].isdigit():
                sIdx = int(args.img_edit_row[0])
                eIdx = int(args.img_edit_row[0]) + 1
            else:
                sIdx = 0
                eIdx = r
        elif len(args.img_edit_row) == 2:
            sIdx = int(args.img_edit_row[0])
            eIdx = int(args.img_edit_row[1])+1
        else:
            raise Exception("Specify a range ")
           
        for i in range(sIdx, eIdx):
            temp_img = np.copy(img)
            temp_img[i, :, :] = 0*temp_img[i, :, :]
            images.append((temp_img, class_label))
            out_img_name.append('row_edit' + str(i))
            
    # Col wise editing of the image
    elif args.img_edit_col is not None:
        print('in col edit')
        img , class_label = images[0]
        (r, c, z) = np.shape(img)
        
        ## Editting coloumns 
        if len(args.img_edit_col) == 1:
            if args.img_edit_col[0].isdigit():
                sIdx = int(args.img_edit_col[0])
                eIdx = int(args.img_edit_col[0]) + 1
            else:
                sIdx = 0
                eIdx = c
        elif len(args.img_edit_col) == 2:
            sIdx = int(args.img_edit_col[0])
            eIdx = int(args.img_edit_col[1])+1
        else:
            raise Exception("Specify a range ")
            
        for i in range(sIdx, eIdx):
            temp_img = np.copy(img)
            temp_img[:, i, :] = 0*temp_img[:, i, :]
            images.append((temp_img, class_label))
            out_img_name.append('col_edit' + str(i))
            
            
    if args.vertical_flip is not None:
        print('Vertical Flip')
        img, class_label = images[0]
        images.append( ( np.flipud(img), class_label ) )
        out_img_name.append('vertical_flip')
        
    if args.horizontal_flip is not None:
        print('Horizontal Flip')
        img, class_label = images[0]
        images.append( ( np.fliplr(img), class_label ) )
        out_img_name.append('horizontal_flip')
        
    if args.rotation_angle is not None:
        print('Rotating the image')
        img, class_label = images[0]
        angle = float(args.rotation_angle)
        rotate_img = ndimage.rotate( img, angle, reshape=False )
        images.append( ( rotate_img, class_label ) )
        out_img_name.append( 'rotate_angle' + str(angle) )
        
    if args.translate_image is not None:
        if len(args.translate_image) > 3:
            raise Exception("Only X-Y translation is allowed. Just provide two values")
        else:
            print('Image Translation')
            temp, class_label = images[0]
            temp = temp.astype(np.float64) # Don't know why am I required to convet it to float64
            tx, ty = args.translate_image
            tform = AffineTransform(translation=(tx, ty))
            t_image = warp(temp, tform.inverse)
            images.append( ( t_image, class_label ) )
            out_img_name.append( 'tx_' + str(tx) + '_ty_' + str(ty) )
    
    # Defining analyzers
    input_range = net["input_range"]
    nscale = (input_range[1]-input_range[0]) * 0.1

    # Methods we use and some properties.
    methods = [
        # NAME                            OPT.PARAMS                 POSTPROC FXN               TITLE
        # Show input.
        ("input",                         {},                        imgnetutils.image,         "Input"),

        # Function
        # ("gradient",                      {"postprocess": "abs"},    imgnetutils.graymap,       "Gradient"), 
        ("gradient",                      {"postprocess": "abs"},    imgnetutils.heatmap,       "Gradient"), # Added (There are post-processing functions in the lib. Just for plotting purposes)
        # ("gradient",                      {"postprocess": "abs"},    imgnetutils.bk_proj,       "Gradient"), # Added
        
        # ("smoothgrad",                   {"augment_by_n": 10,
                                         # "noise_scale": nscale,
                                         # "postprocess": "square"}, imgnetutils.graymap,       "SmoothGrad"),
                                         
        ("smoothgrad",                   {"augment_by_n": 10,
                                         "noise_scale": nscale,
                                         "postprocess": "square"}, imgnetutils.heatmap,       "SmoothGrad"), #Added
                                         
                                          
        # ("integrated_gradients",         {"steps": 10,
                                         # "postprocess": "abs"},    imgnetutils.graymap,       "Integrated Gradients"),
                                         
        ("integrated_gradients",         {"steps": 10,
                                         "postprocess": "abs"},    imgnetutils.heatmap,       "Integrated Gradients"), # Added
                                         

        # Signal
        # ("deconvnet",                    {},                        imgnetutils.bk_proj,       "Deconvnet"),
        ("deconvnet",                    {},                        imgnetutils.heatmap,       "Deconvnet"),
        
        #("guided_backprop",               {},                        imgnetutils.graymap,       "Guided Backprop",), # Added
        ("guided_backprop",               {},                        imgnetutils.heatmap,       "Guided Backprop",), # Added
        # ("guided_backprop",               {},                        imgnetutils.bk_proj,       "Guided Backprop",), 
        
    
        # Interaction
        ("input_t_gradient",              {},                        imgnetutils.heatmap,       "Input * Gradient"), # input*gradients 
        
        #("lrp.z",                         {},                        imgnetutils.graymap,       "LRP-Z"), # Added
        ("lrp.z",                         {},                        imgnetutils.heatmap,       "LRP-Z"), # Checking this
        #("lrp.z",                         {},                        imgnetutils.bk_proj,       "LRP-Z"), # Added
        
        ("lrp.epsilon",                   {"epsilon": 1},            imgnetutils.heatmap,       "LRP-Epsilon"), # Checking this
        
        ("lrp.sequential_preset_a_flat",  {"epsilon": 1},            imgnetutils.heatmap,       "LRP-PresetAFlat"), # Checking this
        #("lrp.sequential_preset_a_flat",  {"epsilon": 1},            imgnetutils.graymap,       "LRP-PresetAFlat"), # Added
        #("lrp.sequential_preset_a_flat",  {"epsilon": 1},            imgnetutils.bk_proj,       "LRP-PresetAFlat"), # Added
        
        ("lrp.sequential_preset_b_flat", {"epsilon": 1},            imgnetutils.heatmap,       "LRP-PresetBFlat"),
    ]
    

    # Create model without trailing softmax
    model_wo_softmax = iutils.keras.graph.model_wo_softmax(model)

    # Create analyzers.
    analyzers = []
    for method in methods:
        try:
            analyzer = innvestigate.create_analyzer(method[0],        # analysis method identifier
                                                    model_wo_softmax, # model without softmax output
                                                    neuron_selection_mode="index",
                                                    **method[1])      # optional analysis parameters
        except innvestigate.NotAnalyzeableModelException:
            # Not all methods work with all models.
            analyzer = None
        analyzers.append(analyzer)
        
    print('Analyzers defined')

    

    # # Running Analysis of the image    
    # analysis = np.zeros([len(images), len(analyzers)]+net["image_shape"]+[3])
    # text = []
    
    # Correct till here
    i = 0
    #(x, y) = images[0]
    heatmap_grids = []
    extra_info = []
    for out_img_name_idx, (x, y) in enumerate(images):
        print('Name of the image currently being innvestigated is: ', out_img_name[out_img_name_idx])
        
        # Add batch axis.
        x = x[None, :, :, :]
        x_pp = imgnetutils.preprocess(x, net)

        # Predict final activations, probabilites, and label.
        presms = model_wo_softmax.predict_on_batch(x_pp)[0] # output is in form of a list, that's why you only take the 0th element
        probs = model.predict_on_batch(x_pp)[0]
        y_hat = probs.argmax()
        #import pdb; pdb.set_trace()
        
        if args.labels == "True":
            sel_neurons = [y]
            tmp_labels = "True"
        elif args.labels[0] == "pred":
            sel_neurons = [y_hat]
            tmp_labels = "pred"
        elif (args.labels[0])[:3] == "top":
            tp = int((args.labels[0])[3:])
            sel_neurons = np.argsort(-presms)[:tp]
            tmp_labels = args.labels[0]
        elif (args.labels[0])[:6] == "bottom":
            tp = int((args.labels[0])[6:])
            sel_neurons = np.argsort(-presms)[-tp:]
            tmp_labels = args.labels[0]
        else:
            sel_neurons = [int(i) for i in args.labels]
            tmp_labels = "user_specified"
            
        #import pdb; pdb.set_trace()
       
            
        for neuron in sel_neurons:
            print('Neuron being clamped', neuron)
            
            analysis = np.zeros([1, len(analyzers)]+net["image_shape"]+[3]) #Computing analysis, one image at a time
            # Save prediction info:
            text = []
            if neuron == 1000:
                text.append(("%s"   % label_to_class_name[y],      # ground truth label
                             "%.2f" % presms.max(),                # pre-softmax logits
                             "%.2f" % probs.max(),                 # probabilistic softmax output  
                             "%s"   % label_to_class_name[y_hat],  # predicted label
                             "%d"   % y_hat,                       #predicted neuron
                             "%d"   % neuron,                      # given neuron
                             "%s"   % label_to_class_name[neuron], # label for given neuron
                             "%.2f" % presms[y_hat],               # pre-softmax logits for given neuron
                             "%.2f" % probs[y_hat]                 # probabilistic softmax output for the given neuron
                            ))
            else:
            
                text.append(("%s"   % label_to_class_name[y],      # ground truth label
                             "%.2f" % presms.max(),                # pre-softmax logits
                             "%.2f" % probs.max(),                 # probabilistic softmax output  
                             "%s"   % label_to_class_name[y_hat],  # predicted label
                             "%d"   % y_hat,                       # predicted_neuron
                             "%d"   % neuron,                      # given neuron
                             "%s"   % label_to_class_name[neuron], # label for given neuron
                             "%.2f" % presms[neuron],              # pre-softmax logits for given neuron
                             "%.2f" % probs[neuron]                # probabilistic softmax output for the given neuron
                            ))
            
            for aidx, analyzer in enumerate(analyzers):
                if methods[aidx][0] == "input":
                    # Do not analyze, but keep not preprocessed input.
                    a = x/255
                    #if out_img_name_idx >= 17:
                    #    import pdb; pdb.set_trace()
                elif analyzer:
                    # Analyze.
                    if neuron == 1000:
                        a = analyzer.analyze(x_pp, neuron_selection = y_hat)
                    else:
                        a = analyzer.analyze(x_pp, neuron_selection = neuron)
                    
                    # Apply common postprocessing, e.g., re-ordering the channels for plotting.
                    a = imgnetutils.postprocess(a, color_conversion, channels_first)
                    # Apply analysis postprocessing, e.g., creating a heatmap.
                    # import ipdb; ipdb.set_trace()
                    a = methods[aidx][2](a) #Apply post-processing on analysis 
                else:
                    a = np.zeros_like(image)
                    
                # Store the analysis.
                analysis[0, aidx] = a[0]
            print('Analysis Done for a neuron')    
            
            #import pdb; pdb.set_trace()


            # Prepare the grid as rectengular list
            grid = [[analysis[i, j] for j in range(analysis.shape[1])]
                    for i in range(analysis.shape[0])]  
            # Prepare the labels
            label, presm, prob, pred, predNeuron, givenNeuron, givenNeuron_label, givenNeuron_logit, givenNeuron_prob = zip(*text)
            row_labels_left = [('tL: {}'.format(label[i]),'pL: {}'.format(pred[i]), 'pN: {}'.format(predNeuron[i]), 'cN: {}'.format(givenNeuron[i]), 'cNL: {}'.format(givenNeuron_label[i])) 
                               for i in range(len(label))]
            row_labels_right = [('mPLog: {}'.format(presm[i]),'mPPr: {}'.format(prob[i]), 'cNLog: {}'.format(givenNeuron_logit[i]), 'cNPr: {}'.format(givenNeuron_prob[i])) 
                                for i in range(len(label))]
            col_labels = [''.join(method[3]) for method in methods]
            
            #import pdb; pdb.set_trace()
            if args.save == 'n':
                 print('not saving the results')
            else:
                eutils.plot_image_grid_final( grid, row_labels_left, row_labels_right, col_labels,
                                       file_name=os.environ.get("plot_file_name", 
                                                                ( args.out_path + imgName.split('.')[0] + '/' + 'experiment_' + timestr + '_' + out_img_name[out_img_name_idx] + '_LABELS_' + tmp_labels +'_neuron' + str(neuron) +'.png' ) 
                                                               )
                                       )
            #print(len(grid))
            #print(len(grid[0]))
            heatmap_grids.append(grid) 
            extra_info.append([row_labels_left, row_labels_right])
    #import ipdb; ipdb.set_trace()
    print('Deleting the memory consuming variables')
    keras.backend.clear_session()
    print('Done')
    return heatmap_grids, extra_info

if __name__ == '__main__':
    args = get_arguments()
    start = time.time()
    main(args)
    print("--- %s seconds ---" % (time.time() - start))