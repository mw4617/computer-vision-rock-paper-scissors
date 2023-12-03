"""
Rock paper scissor game script

This script contains the rock paper scissors game, it does not work with camera input, but rather with manual user input, where user responds to command prompt asking them to make their choice

"""

import random as rand 

def play():

    """ 

    Plays the game.

    This void function runs the game, by calling get_winner()

    """
    get_winner()

def get_winner():
    
    """ 
    
    Determines who won.

    Void function determines if user or computer won by calling get_user_choice() and get_computer_choice()
    
    """

    user_choice=get_user_choice()

    computer_choice=get_computer_choice()
    
    if user_choice is computer_choice:

        print(f"It is a tie! Computer's choice is {computer_choice.lower()}.")        

    elif (user_choice is "Rock" and computer_choice is "Scissors") or (user_choice is "Paper" and computer_choice is "Rock") or (user_choice is "Scissors" and computer_choice is "Paper"):

        print(f"You won! Computer's choice is {computer_choice.lower()}.")   

    else:

        print(f"You lost. Computer's choice is {computer_choice.lower()}.")    


def get_user_choice(): 

    """ 
    
    Determines user selection.

    This function deterimines if user chose rock, paper or scissors based input from the user - r, p, s

    Returns: "Rock", "Paper" or "Scissors" string
    
    """

    letter_key_mapping_dict={"r":"Rock","p":"Paper","s":"Scissors"} #Maps user input to key

    user_input="NA" #intialising user input

    while user_input not in {"r","p","s"}:
        
        user_input=(input("Type letter, r for Rock, p for Paper, s for Scissors: ")).strip().lower() # user input inform of lower case letter

    return letter_key_mapping_dict[user_input] #Returning user choice in form of key "Rock", "Paper" or "Scissors"   

def get_computer_choice():

    """
    
    Determines computers selection

    Determines computers selection by generating random number 1-3 and the classyifng answer as either "Rock", "Paper" or "Scissors"

    Returns: "Rock", "Paper" or "Scissors" string
    
    """

    number_key_mapping_dict={1:"Rock",2:"Paper",3:"Scissors"} #Maps number input to key

    computer_input=rand.randint(1,3) #random number between 1-3 same key as user input

    return number_key_mapping_dict[computer_input] #Returning user computer in form of key "Rock", "Paper" or "Scissors"  

play() #calling the play function to play the game

