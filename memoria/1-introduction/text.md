# Introduction

## Project Background
In recent years, deep learning (DL) has significantly improved the performance of computer
vision systems in image classification, object detection and segmentation tasks.  Generative adversarial networks (GAN) in particular have revolutionized generative tasks like image synthesis
and image-to-image translation.
Image-to-image translation is the problem where we have a source image and we want to generate an image based on the source but with different characteristics.

The project is a continuation of a work [1] where the idea of image-to-image translation is applied to medical imaging, specifically to generation of histological images with the appearance of H&E stained slides using a confocal microscopy (CM) as the source.

Confocal microscopy has enabled rapid evaluation of tissue samples directly in the surgery room
significantly reducing the time of complex surgical operations in skin cancer, but the output largely differs from the standard H&E slides that pathologists use to analyze tissue samples.

Confocal microscopes can operate in two modes which highlight different microscopic structures in the tissue.

In the previous work, a deep learning technique is proposed to combine the two modes of the CM into a H&E-like image to facilitate their interpretation by untrained pathologists and surgeons.

The initial chosen architectures are: a CNN with a multiplicative residual connection to reduce the noise and then a CycleGan to combine the two CM modes into a digitally stained slide.

The current model has room for improvements: the despeckling neural network sometimes produces undesired artifacts and de stain generative network can create or eliminate structures on the output image.
This project will focus on understanding why this happens and will try to solve this problems.

## Need

## Problem specification
Image-to-image translation

## Techniques

## Work Structure
