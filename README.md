# barrel
For barrel cortex modelling

## Compiling the examples
For compiling the examples, you need to have:

- `cmake`
- `boost`, for parsing the config files
- `bullet`, we use this as physic engine

### Using cmake

For compiling: `cmake -D BULLET_PHYSICS_SOURCE_DIR:SPRING=/path/to/your/bullet/repo/ /path/to/your/repo/bullet_demos_extracted/examples/`. Currently, only linux and mac is supported. And bullet should be installed using sudo.

### Parameters for running the examples

Once compiled and built, change directory to `/path/to/your/build/ExampleBrowser/` and then run `./App_ExampleBrowser --config_filename=configFileName --mp4=outputVideoFileName`.
