# Copyart

## How to install dataset

        make install

you will probably get forced to add an api token from your kaggle account to your home .kaggle

---

You must install pytorch (https://pytorch.org/get-started/locally/)

I currently use torch version 1.11.0 with cuda 11.3

## Download pretrained model

        ./scripts/download_model.sh

## How to run CLI

        ./copyart.py -i <image_path> -a <artist_name> [-o <output_path>]

## How to run web app

        python3 app/views.py

        # go to http://localhost:5000/

## Think to look

- Resnet

- PatchGAN

- Adam optimizer