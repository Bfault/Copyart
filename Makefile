NAME	:=	Copyart

DATASET_ARTWORKS_AUTHOR	:=	ikarus777
DATASET_ARTWORKS_NAME	:=	best-artworks-of-all-time
DATASET_FLICKR_AUTHOR	:=	adityajn105
DATASET_FLICKR_NAME		:=	flickr8k
DATASET_PATH			:=	./datasets

PATH := /home/$(USER)/.local/bin:$(PATH)

.SILENT:

$(NAME):
	cat $(NAME)

all: $(NAME)

install:
	pip install --user -r requirements.txt
	kaggle datasets download $(DATASET_ARTWORKS_AUTHOR)/$(DATASET_ARTWORKS_NAME) -p $(DATASET_PATH)
	kaggle datasets download $(DATASET_FLICKR_AUTHOR)/$(DATASET_FLICKR_NAME) -p $(DATASET_PATH)
	unzip $(DATASET_PATH)/$(DATASET_ARTWORKS_NAME).zip -d $(DATASET_PATH)/artworks
	unzip $(DATASET_PATH)/$(DATASET_FLICKR_NAME).zip -d $(DATASET_PATH)/flickr
	rm $(DATASET_PATH)/$(DATASET_ARTWORKS_NAME).zip
	rm $(DATASET_PATH)/$(DATASET_FLICKR_NAME).zip

clean_dataset:
	rm -rf data

clean:
	rm -rf $(NAME)

fclean: clean
	rm -f $(NAME)

re: fclean all