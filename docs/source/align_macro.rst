ImageJ macro for processing experiment projects
===============================================

.. _imagej_macro:


You can apply custom preprocessing steps to stacks organized in a Celldetective experiment project using an ImageJ macro. The provided macro automatically detects the well/position structure within the experiment folder.

Inside the innermost loop (after opening a stack), you can define your preprocessing steps. When running the macro, you will be prompted to select the root folder of your Celldetective experiment, and the macro will handle the folder traversal.

Registration with SIFT: Example Macro
-------------------------------------

Below is an example macro for stack registration using the Linear Stack Alignment with SIFT Multichannel plugin in ImageJ. This macro aligns movies based on a target channel and saves the output with a specific prefix (``Aligned_``).


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
                        
                        // Open stack
                        open(experiment+well+pos+"movie"+File.separator+movie[k]);
                        Stack.setDisplayMode("grayscale");

                        // Here write the preprocessing steps
                        Stack.setChannel(target_channel_int);
                        run("Enhance Contrast", "saturated=0.35");
                        run("Linear Stack Alignment with SIFT MultiChannel", "registration_channel="+target_channel+" initial_gaussian_blur=1.60 steps_per_scale_octave="+octave_steps+" minimum_image_size=64 maximum_image_size=1024 feature_descriptor_size=4 feature_descriptor_orientation_bins=8 closest/next_closest_ratio=0.92 maximal_alignment_error=25 inlier_ratio=0.05 expected_transformation=Rigid interpolate");

                        // Save output
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