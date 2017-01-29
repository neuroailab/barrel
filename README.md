# barrel
For barrel cortex modelling

## Compiling the examples
For compiling the examples, you need to have:

- `cmake`
- `boost`, for parsing the config files
- `bullet`, we use this as physic engine. We also required that the build of bullet is done in "bullet\_build" under bullet source code directory.

### Using cmake

For compiling: `cmake -D BULLET_PHYSICS_SOURCE_DIR:SPRING=/path/to/your/bullet/repo/ /path/to/your/repo/bullet_demos_extracted/examples/`. Currently, only linux and mac is supported. And bullet should be installed using sudo. If your bullet and boost is installed locally, then you also need to specify the "BOOST\_ROOT" and "BULLET\_ROOT" by "-D BOOST\_ROOT:SPRING=/path/to/your/boost/".

### Parameters for running the examples

Once compiled and built, change directory to `/path/to/your/build/ExampleBrowser/` and then run `./App_ExampleBrowser --config_filename=configFileName --mp4=outputVideoFileName`.
