# AutoGraph Spiral

## Description
A script that generates AutoGraph pencil plotter gcode in a archimedes spiral pattern. Based on a source image.

## CONOPS
1. User inputs reference image, drawing dimensions (in milimeters), and sprial parameters.
2. System outputs a AutoGraph compatible gcode file.

## Development Steps
1. ~~generate spiral segments and display as svg~~
2. ~~clip spiral coverage to image boundaries~~
3. ~~create segment centers~~
4. ~~create spiral as large as input image~~
5. ~~move center of spiral~~
6. determine the number of turns required to achieve image coverage
7. create segement centers and nearest pixel center for each center
8. compare spiral segment centers to source image pixel colors
9. generate gcode and inspect in gcode viewer