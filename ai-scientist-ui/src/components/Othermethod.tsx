import React, { useState } from 'react';
import { Button } from '@mui/material';
import axios from 'axios';

const Othermethod: React.FC = () => {
  const [hasFetchedData, setHasFetchedData] = useState(false);
  const [imagePaths, setImagePaths] = useState<any>(null);

  const openNewWindow = async () => {
    const newWindow = window.open('', '_blank', 'width=1024,height=768,scrollbars=yes');

    if (newWindow) {
      newWindow.document.write(`
        <html>
          <head>
            <title>UMAP Plots</title>
            <style>
              body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                text-align: center;
                background-color: #f4f4f4;
              }
              h2 {
                margin-top: 20px;
                color: #333;
              }
              .spinner {
                border: 16px solid #f3f3f3;
                border-top: 16px solid #3498db;
                border-radius: 50%;
                width: 120px;
                height: 120px;
                animation: spin 2s linear infinite;
                margin: 50px auto;
                display: block;
              }
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
              #content {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding-bottom: 50px;
                overflow: auto;
              }
              img {
                display: block;
                margin-bottom: 30px;
                max-width: 95%; /* Ensure the image is responsive */
                height: auto; /* Maintain aspect ratio */
                object-fit: contain; /* Preserve aspect ratio and fit within the dimensions */
                border: 2px solid #ccc;
                border-radius: 8px;
              }
              img:first-of-type {
                padding-top: 0px; /* Large padding for the first image */
              }
            </style>
          </head>
          <body>
            <h2>UMAP Plots</h2>
            <div id="content">
              <div class="spinner"></div>
            </div>
          </body>
        </html>
      `);

      if (hasFetchedData && imagePaths) {
        let imagesHTML = '';
        imagesHTML += `<img src="http://localhost:5000/umap/${encodeURIComponent(imagePaths.celltypist_predicted_labels)}" alt="UMAP Plot 1">`;
        imagesHTML += `<img src="http://localhost:5000/umap/${encodeURIComponent(imagePaths.celltypist_majority_voting)}" alt="UMAP Plot 2">`;
        imagesHTML += `<img src="http://localhost:5000/umap/${encodeURIComponent(imagePaths.gptcelltype_predict)}" alt="UMAP Plot 3">`;

        newWindow.document.getElementById('content')!.innerHTML = imagesHTML;
      } else {
        try {
          const response = await axios.get('http://localhost:5000/other_methods');
          if (response.status === 200) {
            setHasFetchedData(true);
            const paths = response.data.image_paths;
            setImagePaths(paths);

            let imagesHTML = '';
            imagesHTML += `<img src="http://localhost:5000/umap/${encodeURIComponent(paths.celltypist_predicted_labels)}" alt="UMAP Plot 1">`;
            imagesHTML += `<img src="http://localhost:5000/umap/${encodeURIComponent(paths.celltypist_majority_voting)}" alt="UMAP Plot 2">`;
            imagesHTML += `<img src="http://localhost:5000/umap/${encodeURIComponent(paths.gptcelltype_predict)}" alt="UMAP Plot 3">`;

            newWindow.document.getElementById('content')!.innerHTML = imagesHTML;
          } else {
            newWindow.document.getElementById('content')!.innerText = 'Error generating UMAP plots.';
          }
        } catch (error) {
          newWindow.document.getElementById('content')!.innerText = 'Error: Unable to fetch UMAP plots.';
          console.error(error);
        }
      }
    }
  };

  return (
    <Button
      variant="contained"
      onClick={openNewWindow}
      sx={{
        backgroundColor: 'white',
        color: 'black',
        fontSize: '12px',
        border: '1px solid black',
        marginBottom: '10px',
        '&:hover': {
          backgroundColor: 'lightgray',
        },
      }}
    >
      Get Other Method
    </Button>
  );
};

export default Othermethod;
