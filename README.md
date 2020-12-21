# Virtual moving sound synthesis

This project synthesizes stereo audio signals that simulate moving sound  sources for a specified angular speed and path at a given resolution

It first extracts the HRIR from the Wisconsin HRTF database, computes the equalization required for diffuse field response, interpolates linearly in frequency domain for the positions requested, resamples to the output sampling rate required, convolves with input of user choice and applies equalization together with global scale factor and then concatenates the responses to stimulate movement in the virtual soundscape and finally saves the output as wav file
