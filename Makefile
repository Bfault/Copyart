NAME	=	Copyart

$(NAME):
	cat $(NAME)

all: $(NAME)

install:
	pip install -r requirements.txt
	export PATH="/home/$USER/.local/bin:$PATH"
	./install_dataset

clean_dataset:
	rm -rf data

clean:
	rm -rf $(NAME)

fclean: clean
	rm -f $(NAME)

re: fclean all