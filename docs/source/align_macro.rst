SIFT alignment with ImageJ macro
================================

.. _align_macro:

We highly recommend that you align the movie beforehand using for example, the "Linear Stack Alignment with SIFT Multichannel" tool available in Fiji, when activating the PTBIOP update site [#]_ (see discussion here_). 

.. _here: https://forum.image.sc/t/registration-of-multi-channel-timelapse-with-linear-stack-alignment-with-sift/50209/16


How to use it
-------------

.. figure:: _static/align-stack-sift.gif
    :align: center
    :alt: sift_align
    
    Demonstration of the of the SIFT multichannel tool on FIJI

We provide an ImageJ macro to perform the registration of batches of ADCC movies. We ask the user to put all of the .tif files to register in a folder and create a subfolder "aligned/" in which the registered files will be saved. 

.. code-block:: java

    run("Collect Garbage");
    experiment = getDirectory("Experiment folder containing movies to align...");
    wells = getFileList(experiment);

    octave_steps = "7";
    target_channel = "4";
    target_channel_int = 4;
    prefix = "After"

    for (i=0;i<wells.length;i++){
        
        well = wells[i];
        
        if(endsWith(well, File.separator)){
            positions = getFileList(experiment+well);
            for (j=0;j<positions.length;j++) {
                pos = positions[j];
                movie = getFileList(experiment+well+pos+"movie"+File.separator);
                for (k=0;k<movie.length;k++) {
                    if (startsWith(movie[k], prefix)) {
                        open(experiment+well+pos+"movie"+File.separator+movie[k]);
                        Stack.setDisplayMode("grayscale");
                        Stack.setChannel(target_channel_int);
                        run("Enhance Contrast", "saturated=0.35");
                        run("Linear Stack Alignment with SIFT MultiChannel", "registration_channel="+target_channel+" initial_gaussian_blur=1.60 steps_per_scale_octave="+octave_steps+" minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Rigid interpolate");
                        saveAs("Tiff", experiment+well+pos+"movie"+File.separator+"Aligned_"+movie[k]);
                        close();
                        close();
                        run("Collect Garbage");
                    }
                }
            }
            
        }
    }

    print("Done.");

This code is a Fiji macro script that performs image alignment on TIFF files located in a specific folder. It prompts the user to select the folder, gets a list of .tif files in the folder and runs image alignment using Linear Stack Alignment with SIFT MultiChannel algorithm on each image. The aligned images are then saved in a subfolder with a prefix "Aligned_" added to the original file name. It also run "Collect Garbage" at the start and end of loop to free up memory.

References
----------

.. [#] https://www.epfl.ch/research/facilities/ptbiop/