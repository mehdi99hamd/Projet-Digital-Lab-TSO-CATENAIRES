# Projet-Digital-Lab-TSO-CATENAIRES

In this project, a point cloud cutting tool was created to :

- Automatic identification of catenary poles.
- Creation of cuts through the poles that are perpendicular to the rails of the the track.
- Automatic naming of the cuts with coordinates.
- Export of the point cloud cuts in DXF or DWG format.



### Model Used : VGG-19

We have chosen the VGG-19 as the basis of our model. The VGG is the abbreviation of "Visual Geometry Group Net". It has been used in several classification problems. It has about 143 million parameters, which were trained using ImageNet, a database with 1.2 million images that contains thousands of classes. It has a very robust architecture used for comparative analysis. The VGG-19 neural network is composed of The neural network VGG-19 is composed of 19 layers of neural networks forming 5 blocks. We used the first four by adding three dense layers.


### Graphic Interface : Dash Plotly 

Dash is a python framework created by plotly to develop interactive web applications. Dash is written on top of Flask, Plotly.js and React.js. With Dash, you don't need to learn advanced HTML, CSS or Javascript to create interactive dashboards. interactive dashboards. Dash is open source and the applications built with this framework are framework are visualized on the web browser.

The objective of our interface is to download the files with the extensions .e57 and .las, to process them and to detect the catenaries present in them and finally to save each pair of catenaries in a single file with the extension .dxf.
