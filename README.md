# Kilter Board application

## What is a Kilter Board?
A Kilter Board is an adjustable training wall for climbers. It can range from 0 down to 70 degrees overhanging and has thousands of climbs available for users to try hrough the use of an app available through the google or apple store that light up the correspnding holds on a bluetooth connected board.
Kilter Boards are becoming more common in commerical gyms and this tool is aimed to help climbers develop novel climbs to assist in training and to remove alot of the trial and error that comes through manually creating climbs.

![image](https://user-images.githubusercontent.com/53500810/211188704-bb1fb035-bbfb-4c9a-b24a-a0a7b83237f4.png)
Credit: ClimbFit Kirrawee https://www.climbfit.com.au/location/climbfit-kirrawee/

This application has two functions. The first is to perform EDA on various climbs on the Kilter Board app aiming to discover patterns and most commonly used hodls across a 
range of difficulties and board angles. 

## EDA 
One of the outputs is the number of climbs at each grade using the v scale as shown below
![image](https://user-images.githubusercontent.com/53500810/211188745-388d5d03-23b0-4778-a43a-4eee875efa82.png)

Another visual presented in the app is an adjustable histogram of the number of climbs for each difficulty as the angle changes as well as the number of recorded ascents for each angle
![image](https://user-images.githubusercontent.com/53500810/211188784-0095bc38-032d-4ea5-ab13-f0d8375896d2.png)

![image](https://user-images.githubusercontent.com/53500810/211188814-64148f52-7637-4140-a159-5e6ad725fd1c.png)

I have also created a heatmap of the most commonly used holds at each grade and angle. 

![image](https://user-images.githubusercontent.com/53500810/211188991-51d3f891-0c09-4d31-a91e-1eb993c543cb.png)

## Climb Creation

A Generative Adverserial Network (GAN) was used to create thousands of new climbs by training on established climbs. The hold sequences were index based encoded as each hold could be used as either a footholds, handhold, start or finish.
Users will see new climbs each time they load as the GAN generates new climbs each time the page is loaded. Users can select the board angle and difficulty grade (v grade). The final slider shows how many climbs generated for those two conditions. Future work would involve using the board angle and grade to generate climbs at those variables.

![image](https://user-images.githubusercontent.com/53500810/211189172-7ce69f5e-82c3-4b04-8152-98921ac6ba40.png)

The quaility of the climbs range significantly with some climbs requiring no adjustment or potentially an additional foothold, to some that have strange off putting movement. It's intented that generated climbs will serve as a starting point for users to build upon.


