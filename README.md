<div align="center">
  <h1>📸 Basic Image Preprocessing 📸</h1>
  
  [![PyPI version](https://img.shields.io/pypi/v/basic_image_preprocessing)](https://pypi.org/project/basic-image-preprocessing/)
  [![Downloads](https://static.pepy.tech/badge/basic-image-preprocessing)](https://pepy.tech/project/basic-image-preprocessing)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/thivagar-manickam/basic-image-preprocessing/blob/main/LICENSE)
  [![Supported Python Versions](https://img.shields.io/pypi/pyversions/basic-image-preprocessing.svg)](https://pypi.org/project/basic-image-preprocessing/)
</div>


Basic Image Preprocessing is a Python package that is focused on handling the various Image enhancement
and Noise removal techniques.

The package encompasses most of the basic methods and algorithms used for enhancing the quality 
of the image as well as the removal of noise from the image.


## Quick Install

You can install the package directly by using the pip command or through the conda command prompt

`pip install basic_image_preprocessing`
or
`conda install basic_image_preprocessing -c conda-forge`


## List of Techniques Available in the Package

<table>
  <thead>
    <tr>
      <td><b>Types</b></td>
      <td><b>Method</b></td>
      <td><b>Sub Types</b></td>
      <td><b>Example File</b></td>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan=3>Traditional</td>
      <td>Linear Equation</td>
      <td></td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Traditional%20Methods.ipynb">Linear Equation Usage Examples</a></td>
    </tr>
    <tr>
      <td>Non Linear Methods</td>
      <td>
        <ol>
          <li>Power transformation</li>
          <li>Exponential transformation</li>
          <li>Lograthimic transformation</li>
        </ol>
      </td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Traditional%20Methods.ipynb">Non Linear Method Usage Examples</a></td>
    </tr>
    <tr>
      <td>Basic Mathematical Operations</td>
      <td>
        <ol>
          <li>Addition</li>
          <li>Subtraction</li>
          <li>Multiplication</li>
          <li>Division</li>
        </ol>
      </td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Traditional%20Methods.ipynb">Basic Mathematical Operation Usage Examples</a></td>
    </tr>
    <tr>
      <td rowspan=2>Conventional</td>
      <td>Histogram Equalization</td>
      <td></td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Conventional%20Methods.ipynb">Histogram Equalization Usage Examples</a></td>
    </tr>
    <tr>
      <td>CLAHE</td>
      <td></td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Conventional%20Methods.ipynb">CLAHE Usage Examples</a></td>
    </tr>
    <tr>
      <td rowspan=3>Edge Detection</td>
      <td>Laplacian</td>
      <td></td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Edge%20Detection.ipynb">Laplacian Usage Examples</a></td>
    </tr>
    <tr>
      <td>Canny Edge Detection</td>
      <td></td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Edge%20Detection.ipynb">Canny Edge Detection Usage Examples</a></td>
    </tr>
    <tr>
      <td>Edge Filtering Techniques</td>
      <td>
        <ol>
          <li>Sharpenning</li>
          <li>Custom Edge Detection</li>
        </ol>
      </td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Edge%20Detection.ipynb">Edge Filtering technique Usage Examples</a></td>
    </tr>
    <tr>
      <td rowspan=1>Frequency Noise Filtering</td>
      <td>Fourier Transform</td>
      <td></td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Frequency%20Noise%20Filtering.ipynb">Fourier Transform Usage Examples</a></td>
    </tr>
    <tr>
      <td rowspan=3>Spatial Noise Filtering</td>
      <td>Bilateral Filter</td>
      <td></td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Spatial%20Noise%20Filtering.ipynb">Bilateral Filter Usage Examples</a></td>
    </tr>
    <tr>
      <td>Wiener Filter</td>
      <td></td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Spatial%20Noise%20Filtering.ipynb">Wiener Filter Usage Examples</a></td>
    </tr>
    <tr>
      <td>Basic Noise Filtering</td>
      <td>
        <ol>
          <li>Mean Filtering</li>
          <li>Median Filtering</li>
          <li>Gaussian Filtering</li>
        </ol>
      </td>
      <td><a href="https://github.com/thivagar-manickam/image_preprocessing/blob/main/basic_image_preprocessing/examples/Basic%20Image%20Processing%20-%20Spatial%20Noise%20Filtering.ipynb">Basic Noise Filtering Usage Examples</a></td>
    </tr>
  </tbody>
</table>


## 💁 Contributing
If you would like to contribute to this project, create a pull request with your changes and provide
a detailed description of the change being done.


## :lady_beetle: Report a bug
If you find a bug or unexpected behavior when using any of the methods, kindly raise an Issue.
Please follow the bug template [here](https://github.com/thivagar-manickam/image_preprocessing/blob/main/.github/ISSUE_TEMPLATE/bug_report.md) while raising the bug, so that it will be
easy for us to analyze and provide a fix for the issue.


## :placard: Request a new Algorithm / Functionality
If you find any method or algorithm missing from the package, please create a feature request under
the Issue section by following the feature request template found [here](https://github.com/thivagar-manickam/image_preprocessing/blob/main/.github/ISSUE_TEMPLATE/feature_request.md) We will
go through the request and do the required works to get the feature ready.
