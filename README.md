# Digital Visual Effect, 2021 Spring
## Image stiching

### Usage

First create the image list as the following format:

```
# [path to image] [focal_length]
test_data/parrington/prtn00.jpg 704.916
test_data/parrington/prtn01.jpg 706.286
test_data/parrington/prtn02.jpg 705.849
...

```
And run the program:

`usage: main.py [-h] file_list output_image`

```
$ python3 code/main.py list.txt output.png
```

Example to generate result.png

```
$ python3 code/main.py image_lists/test4.txt result.png
```
