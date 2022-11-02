# CS7643

## Project Proposal

https://mega.nz/file/yaZlGL6D#l2_lojZHcMVXZelzbERsXzLVxl6cy-ElY5sRyWXkRP4

## Git workflow

1. pull main
2. make branch with name feature/<what-you're-working-on>
3. PR feature branch to main
4. get at least two approvals?
5. merge

## Setup instructions
1. git clone git@github.com:rreube3/CS7643.git
2. conda create env -f environment.yml
3. conda activate cs7643-final
4. run test-env.py

## Project structure

--> model // Store our Unet and GANs  
--> utils // Training / Testing utilities  
--> img-transform // Image augmentations  
--> barlow // BT implementation  
--> weights // Place to store weights  
--> README.md  
--> environment.yml  
--> notebooks/ 