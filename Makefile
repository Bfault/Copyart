NAME	:=	Copyart

DATASET_AUTHOR	:=	ikarus777
DATASET_NAME	:=	best-artworks-of-all-time
DATASET_PATH	:=	./data

PATH := /home/$(USER)/.local/bin:$(PATH)

.SILENT:

$(NAME):
	cat $(NAME)

all: $(NAME)

install:
	pip install --user -r requirements.txt
	kaggle datasets download $(DATASET_AUTHOR)/$(DATASET_NAME) -p $(DATASET_PATH)
	unzip $(DATASET_PATH)/$(DATASET_NAME).zip -d $(DATASET_PATH)
	rm $(DATASET_PATH)/$(DATASET_NAME).zip

clean_dataset:
	rm -rf data

clean:
	rm -rf $(NAME)

fclean: clean
	rm -f $(NAME)

re: fclean all