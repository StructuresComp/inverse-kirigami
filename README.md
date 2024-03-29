# inverse-kirigami
This repository contains the code for the paper:
- [Rapid design of fully soft deployable structures via kirigami cuts and active learning](https://arxiv.org/abs/2203.11546)

Here, we propose a rapid design approach for fully soft structures that can achieve targeted 3D shapes through a fabrication process that happens entirely on a 2D plane. We develop a symmetry-constrained active learning approach to learn how to explore the large design space efficiently. The proposed framework can accelerate the adoption of morphing structures in a range of areas including soft robotics, additive manufacturing, and the construction industry.

`cnntrain_vae.py` performs variational autoencoder to reduce the dimension of candidate kirigami patterns to low-dimension representations. 

`cnntrainv2_baysesskit.py` performs Bayesian optimization. The "gp_minimize" function (a function in scikit-optimize) conducts Bayesian optimization using Gaussian Processes. 
In other words, the function values are assumed to follow a multivariate Gaussian. The covariance of the function values is given by a GP kernel between the parameters. Then a smart choice to choose the next parameter to evaluate can be made by the acquisition function over the Gaussian prior which is much quicker to evaluate.
The "black_box_functionrt" function is used to compare the 3D deformation pattern predicted by finite element simulation and the desired shape. 


## Requirements
To run the code, you must install the following dependencies first:
- PyTorch (1.8.1)
- [atomai](https://github.com/pycroscopy/atomai)
- [scikit-optimize](https://scikit-optimize.github.io/stable/)

You also need to install softwares:
- [ABAQUS (Finite Element Analysis)] (https://www.3ds.com/products-services/simulia/products/abaqus/)
- [MATLAB] (https://www.mathworks.com/products/matlab.html)

## Files
- `cnntrain_vae.py` performs variational autoencoder to reduce the dimension of candidate kirigami patterns to low dimension representations
- `cnntrainv2_baysesskit.py` performs Bayesian optimization. [The python code start and stop MATLAB Engine and ABAQUS during each round of iteration](https://www.mathworks.com/help/matlab/matlab_external/start-the-matlab-engine-for-python.html).
- Matlab files are used to create the mesh, boundary conditions input to the ABAQUS simulation
- 'readodb.py' reads the .odb files generated by ABAQUS.

## Citation
If you use this code for part of your project or paper, or get inspired by the associated paper, please cite:  

    @misc{Ma2022Rapid,
        doi = {10.48550/ARXIV.2203.11546},
        url = {https://arxiv.org/abs/2203.11546},
        author = {Ma, Leixin and Mungekar, Mrunmayi and Roychowdhury, Vwani and Jawed, M. Khalid},
        keywords = {Computational Physics (physics.comp-ph), FOS: Physical sciences, FOS: Physical sciences},
        title = {Rapid design of fully soft deployable structures via kirigami cuts and active learning},
        publisher = {arXiv},
        year = {2022},
        copyright = {Creative Commons Attribution Non Commercial Share Alike 4.0 International}
    }
