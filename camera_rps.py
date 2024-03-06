import cv2
import sys
import numpy as np
import random as rand 
import click
from keras.models import load_model
import time as clock


class RockPaperScissors:

   '''
   This class is used to represent a game of rock paper and scissors. A human user plays against the AI. The class contains all the methods necessary 
   in order to control the user input through a camera, classify the user's hand shape in to rock paper or scissors and generate a computer response to user and display that to the user.
   In order to achieve greater degree of accuracy there is a separate AI model which classifies if the camera input is in daylight or in artificial lighting setting. This allows to 
   to make slight adjustments to model which classifies the shape depending on lighting conditions, as it was found there was a slight bias towards some shapes that varied with lighting
   conditions. 
   
   Attributes:

        no_of_rounds (int): number of rounds.

        user_wins (int): number of game rounds user won.

        computer_wins (int): number of game rounds computer won.

        lighting_conditions_type (int): Represents the type of lighting detected by the lighting model, affecting prediction accuracy.

        lighting_model (Model from Keras): The pre-trained machine learning model used to classify the lighting conditions.

        model, model_2, model_3, model_4, model_5 (Model from Keras): Models used for predicting hand gestures representing rock, paper, or scissors.

        cap (cv2.VideoCapture): Object for capturing video frames from the camera.

        data (numpy.ndarray): A NumPy array structured to hold pre-processed camera frames for model predictions.

        converged (bool): Indicates whether the prediction models have converged on a single prediction.

        count (int): Used for counting frames or iterations in certain processes.

        text (str): Text to be displayed on the screen, often for messages about model convergence or game outcomes.

        frequency_list (list[int]): Stores frequencies of predictions to help determine convergence.

        timer_started (bool): Indicates whether a timer for user aids or messages has started.

        frame (numpy.ndarray): The current video frame captured from the camera.

        start_time, end_time (float): Track the duration of certain operations or for displaying messages for a set time.

        aid_text_start_time (float): Could serve as a timestamp for when user aid text begins to be displayed, if implemented.

        index_list (list[int]): Stores indices of the maximum prediction probabilities from each model.

        max_index (int): The index of the most frequently predicted gesture among all models.

        paper_anti_bias_factor, scissors_anti_bias_factor, stone_anti_bias_factor (float): Factors to adjust prediction probabilities of each gesture to counteract bias.

        round_outcome (str): Stores the outcome of the current game round.

        game_outcome (str): Stores the final outcome of the game after all rounds are played.

        number_key_mapping_dict (dict[int, str]): Maps numeric predictions to their corresponding gestures ("Rock", "Scissors", "Paper", "Nothing").
   
   '''
   
   def __init__(self, no_of_rounds=None, lighting_condition='unspecified'):

      
      '''
      Initialize the RockPaperScissors class with settings for playing the game.
      This includes setting up the camera, loading prediction models, and initializing game settings.
      
      Parameters:
         no_of_rounds (int, optional): Specifies the total number of rounds to be played in the game.
                                       Defaults to None, which can be set to a default value later.

         lighting_condition (string, optional): Specifies the light condtions in the room, daylight or artificial light.
                                                Defaults to unspecified, in this case AI model is used to predict lighting conditions in the room.                              
      '''
      # Initialise the lighting conditions.
      self.lighting_condition = lighting_condition
      
      # Initialise the total number of rounds for the game. This can be set by the user.
      self.no_of_rounds = no_of_rounds
      
      # Initialise counters for the number of wins by the user and the computer.
      self.user_wins = 0
      self.computer_wins = 0
      
      # Mapping from numerical predictions to their corresponding gestures.
      # This helps in translating model predictions into game moves.
      self.number_key_mapping_dict = {0: "Rock", 1: "Scissors", 2: "Paper", 3: "Nothing"}
      
      # Load pre-trained Keras models for gesture recognition.
      # Each model may be trained under different conditions or configurations to improve accuracy.
      self.model = load_model('keras_model.h5')
      self.model_2 = load_model('keras_model_2.h5')
      self.model_3 = load_model('keras_model_3.h5')
      self.model_4 = load_model('keras_model_4.h5')
      self.model_5 = load_model('keras_model_5.h5')
      
      # Load a pre-trained model specifically for detecting lighting conditions.
      self.lighting_model = load_model('keras_model_lighting_2.h5')
      
      # Set up the camera for capturing video. 0 indicates the first camera device.
      self.cap = cv2.VideoCapture(0)
      
      # Initialize a data structure for storing camera frames formatted for model prediction.
      self.data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
      
      # Variables to manage the state of model predictions and game logic.
      self.converged = False  # Indicates whether predictions from models have converged to a consistent result.
      self.count = 0  # General purpose counter, useful for iterations and controlling flow in the game.
      self.text = "Not Converged"  # Text to display on the UI regarding the state of model convergence.
      self.frequency_list = []  # List to track the frequency of predictions across models for determining convergence.
      
      # Flags and timing for managing user assistance messages.
      self.timer_started = False
      self.frame = None  # To hold the current frame captured from the camera.
      self.start_time = 0
      self.end_time = 0
      self.aid_text_start_time = None  # Specific start time for displaying user aid text, if implemented.
      
      # Lists and variables for managing and interpreting model predictions.
      self.index_list = [0, 0, 0, 0, 0]  # To store indices of predictions from each model.
      self.max_index = 0  # The most frequently occurring index among predictions, representing the consensus gesture.
      
      # Factors to adjust the likelihood of each gesture based on observed biases.
      # These factors are applied to model predictions to mitigate bias.
      self.paper_anti_bias_factor = 1
      self.scissors_anti_bias_factor = 1
      self.stone_anti_bias_factor = 1
      
      # Strings to store the outcomes of individual rounds and the overall game.
      self.round_outcome = ""
      self.game_outcome = ""
      
      # if lighting conditions have been specified by the user
      if self.lighting_condition!='unspecified':
          
          if self.lighting_condition == 'daylight':

               self.lighting_conditions_type=0

          elif self.lighting_condition == 'artificial':

               self.lighting_conditions_type=1

      # if lighting condtions have not been specified by the user, attempt to determine the conditions
               
      else:

         # Preliminary step to determine lighting conditions which might affect prediction accuracy.
         # This loop captures several frames to make an initial assessment of the lighting environment.
         i = 1
         lighting_type = 0
         while i < 5:
            # Capture a frame from the camera.
            ret, self.frame = self.cap.read()
            
            # Resize the frame to the input size expected by the models.
            resized_frame = cv2.resize(self.frame, (224, 224), interpolation=cv2.INTER_AREA)
            
            # Convert the frame to a format suitable for model prediction.
            image_np = np.array(resized_frame)
            normalized_image = (image_np.astype(np.float32) / 127.0) - 1  # Normalize the image.
            self.data[0] = normalized_image
            
            # Predict lighting conditions using the dedicated model.
            self.lighting_prediction = self.lighting_model.predict(self.data)
            
            # Process the lighting prediction to adjust for known biases.
            self.lighting_probabilities = [value for value in self.lighting_prediction[0]]
            self.lighting_probabilities[1] = self.lighting_probabilities[1] / 1.1  # Adjust for artificial light bias.
            self.lighting_probabilities[2] = self.lighting_probabilities[2]/10  # Adjust for sunlight bias.
            
            # Determine the most likely lighting condition.
            self.lighting_conditions_type = self.lighting_probabilities.index(max(self.lighting_probabilities))
            lighting_type += self.lighting_conditions_type
            
            i += 1
            print("Lighting type", self.lighting_conditions_type)
         
         # Finalize the initial lighting condition assessment based on captured frames.
         self.lighting_conditions_type = int(round(lighting_type / i))

   def display_game_rounds(self):
            
      '''
      Display the current game round information and the outcome of the last round on the webcam feed.
      This method updates the video feed to show messages about the number of rounds left and the result of the last round played.
      '''

      # Start timing for message display. This can help in controlling how long messages are shown on the screen.
      start_time = clock.time()
      
      # Enter a loop to continuously update and display the frame with the game information.
      while True:
         # Capture a frame from the camera.
         ret, frame = self.cap.read()

         # Resize the frame for consistency. This step is optional and can be adjusted based on the desired display size.
         resized_frame = cv2.resize(self.frame, (224, 224), interpolation=cv2.INTER_AREA)

         # Check if the frame was successfully captured. If not, print an error message and break from the loop.
         if not ret:
               print("Failed to grab frame")
               break

         # Calculate the size and position for the text to be displayed on the frame.
         # This ensures that the text is centered and properly sized relative to the frame dimensions.
         text_1_size = cv2.getTextSize(f"{self.round_outcome}", cv2.FONT_HERSHEY_SIMPLEX, 0.74, 2)[0]
         text_2_size = cv2.getTextSize(f"Number of game rounds left: {self.no_of_rounds}", cv2.FONT_HERSHEY_SIMPLEX, 0.74, 2)[0]
         x_1 = int((self.frame.shape[1] - text_1_size[0]) / 2)
         y_1 = int((self.frame.shape[0] - text_1_size[1]) / 2 - 30)
         x_2 = int((self.frame.shape[1] - text_2_size[0]) / 2)
         y_2 = int((self.frame.shape[0] - text_2_size[1]) / 2 + 30)
         
         # Draw the text onto the frame. The text displays the outcome of the last round and the number of rounds left.
         cv2.putText(self.frame, f"{self.round_outcome}", (x_1, y_1),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.74, (147, 20, 255), 2)
         cv2.putText(self.frame, f"Number of game rounds left: {self.no_of_rounds}", (x_2, y_2),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.74, (147, 20, 255), 2)

         # Display the updated frame with the text information.
         cv2.imshow('frame', self.frame)

         # Check for a key press to break the loop (e.g., 'x' to exit) or automatically break after 1.5 seconds.
         if cv2.waitKey(1) & 0xFF == ord('x') or clock.time() - start_time > 1.5:
               break
         
         # Pause for a few seconds to ensure the message is readable by the user.
         # This delay can be adjusted based on the desired user experience.
         clock.sleep(5)

      # Release the video capture object and close any OpenCV windows to clean up resources.
      # This is necessary to prevent resource leaks and ensure the application can continue to run smoothly.
      self.cap.release()
      cv2.destroyAllWindows()

      # Reinitialize the capture for the next round.
      # This step is necessary to continue capturing video frames for subsequent rounds.
      self.cap = cv2.VideoCapture(0)

   def display_game_end_result(self):
      '''
      Display the final outcome of the game on the webcam feed.
      This method shows a message indicating whether the user won or lost the game and the final scores.
      '''
      
      # Start timing for message display. This allows for controlling the display duration of the final message.
      start_time = clock.time()
      
      # Continuously update and display the frame with the game's final outcome.
      while True:
         # Attempt to capture a frame from the camera.
         ret, frame = self.cap.read()

         # Check if the frame was successfully captured. If not, print an error message and exit the loop.
         if not ret:
            print("Failed to grab frame")
            break

         # Draw a filled rectangle to "clear" the region where text will be displayed.
         # This ensures that the text is easily readable against the background.
         cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), thickness=cv2.FILLED)

         # Prepare the text for display: the game's final outcome and the score.
         text_1 = f"{self.game_outcome}"
         text_2 = f"Your wins: {self.user_wins} Computer wins: {self.computer_wins}"

         # Calculate the position for each line of text to ensure they are centered on the frame.
         text_1_size = cv2.getTextSize(text_1, cv2.FONT_HERSHEY_SIMPLEX, 0.74, 2)[0]
         x_1 = int((frame.shape[1] - text_1_size[0]) / 2)
         y_1 = int((frame.shape[0] / 2) - 20)

         text_2_size = cv2.getTextSize(text_2, cv2.FONT_HERSHEY_SIMPLEX, 0.74, 2)[0]
         x_2 = int((frame.shape[1] - text_2_size[0]) / 2)
         y_2 = int((frame.shape[0] / 2) + 20)

         # Draw the text on top of the rectangle, effectively displaying the final game outcome and scores.
         cv2.putText(frame, text_1, (x_1, y_1), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (10, 252, 252), 2)
         cv2.putText(frame, text_2, (x_2, y_2), cv2.FONT_HERSHEY_SIMPLEX, 0.74, (10, 252, 252), 2)

         # Display the frame with the end game results.
         cv2.imshow('frame', frame)

         # Check for a key press to exit (e.g., 'x') or exit automatically after 1.5 seconds.
         # This gives the user time to read the final outcome before the window closes.
         if cv2.waitKey(1) & 0xFF == ord('x') or clock.time() - start_time > 1.5:
            break
         
         # Pause for a few seconds before closing the display to ensure the user has time to read the message.
         clock.sleep(7)

      # Release the video capture object and close any OpenCV windows to clean up resources after displaying the message.
      self.cap.release()
      cv2.destroyAllWindows()

   def get_prediction(self):
      
      '''
         This function uses the webcam to  call each of the 5 models to make predictions of the hand shape presented by the user. The classification is expressed inform of the number for each model. 
         Most common index get's chosen. Before the predictions anti-bias factors are applied models according to lighting condtions to fine tune the models and improve accuracy.

         Returns:
            max_index (int): most common of model predicted indices corresponding to the shape number (0 rock 1 scissors 2 paper 3 nothing)
      '''
      
       # Initially, we're not tracking convergence; this will change once the user confirms their gesture.
      tracking_convergence=False 
      
      # Loop indefinitely until a stable prediction is made and confirmed by the user.
      while True: 

         # Capture the next video frame.
         ret, self.frame = self.cap.read()
         
         # Resize the frame to the size expected by the model (224x224 pixels).
         resized_frame = cv2.resize(self.frame, (224,224), interpolation = cv2.INTER_AREA)

         # Convert the resized frame to a numpy array and normalize it for the model.
         image_np = np.array(resized_frame)

         normalized_image = (image_np.astype(np.float32) / 127.0) - 1 # Normalize the image

         self.data[0] = normalized_image

         # Obtain a prediction from each model for the current frame.

         prediction = self.model.predict(self.data)

         prediction_2=self.model_2.predict(self.data)

         prediction_3=self.model_3.predict(self.data)

         prediction_4=self.model_4.predict(self.data)

         prediction_5=self.model_5.predict(self.data)

         # Show the frame in a window to let the user see what the camera is capturing.

         cv2.imshow('frame', self.frame)

         
         # Extract prediction probabilities for each class from the models' outputs.

         probabilites = [value for value in prediction[0]]
         probabilites_2 = [value for value in prediction_2[0]]
         probabilites_3 = [value for value in prediction_3[0]]
         probabilites_4 = [value for value in prediction_4[0]]

         probabilites_5 = [value for value in prediction_5[0]]

         # The fifth model has a different order of labels; adjust them to match the order used by other models.

         rock=probabilites_5[1]

         paper=probabilites_5[0]

         scissors=probabilites_5[2]

         probabilites_5=[rock,scissors,paper] # Corrected order 

         # Apply adjustments to the prediction probabilities based on the detected lighting conditions.


         if(self.lighting_conditions_type==0): # Condition for daylight without direct sunlight

            # Adjust the anti-bias factors for each gesture under these lighting conditions.
            
            self.paper_anti_bias_factor=1000#(probabilites[2])**(-0.87)

            self.scissors_anti_bias_factor=1#(probabilites[1])**(-0.62)

            self.stone_anti_bias_factor=1#(probabilites[0])**(2)

            print("daylight")

         elif(self.lighting_conditions_type==1):  # Condition for artificial light
            
            # Set anti-bias factors for artificial lighting conditions.

            self.paper_anti_bias_factor=1

            self.scissors_anti_bias_factor=10**(-4.75)

            self.stone_anti_bias_factor=1

            print("artificial light")

         else:
            # Handle unplayable lighting conditions, such as direct sunlight causing glare.

            print("Game is not playable with sun in the background / sun light plumes in the background. Please make sure blinds cover all the sunplumes.")   

            sys.exit(0) # Terminate the game due to unplayable lighting conditions.

         # Apply the anti-bias factors to adjust the probabilities.
         # This is done to counteract biases observed in the models' predictions under different lighting conditions.
            
         probabilites[2]=probabilites[2]*self.paper_anti_bias_factor

         probabilites_2[2]=probabilites_2[2]*self.paper_anti_bias_factor

         probabilites_3[2]=probabilites_3[2]*self.paper_anti_bias_factor

         probabilites_4[2]=probabilites_4[2]*self.paper_anti_bias_factor

         probabilites_5[2]=probabilites_5[2]*self.paper_anti_bias_factor

         probabilites[1]=probabilites[1]*self.scissors_anti_bias_factor

         probabilites_2[1]=probabilites_2[1]*self.scissors_anti_bias_factor

         probabilites_3[1]=probabilites_3[1]*self.scissors_anti_bias_factor

         probabilites_4[1]=probabilites_4[1]*self.scissors_anti_bias_factor

         probabilites_5[1]=probabilites_5[1]*self.scissors_anti_bias_factor

         probabilites[0]=probabilites[0]*self.stone_anti_bias_factor

         probabilites_2[0]=probabilites_2[0]*self.stone_anti_bias_factor

         probabilites_3[0]=probabilites_3[0]*self.stone_anti_bias_factor

         probabilites_4[0]=probabilites_4[0]*self.stone_anti_bias_factor

         probabilites_5[0]=probabilites_5[0]*self.stone_anti_bias_factor

         # Calculate the index (0, 1, 2, or 3) with the highest probability from each prediction.

         max_index_0=probabilites.index(max(probabilites))

         max_index_1=probabilites_2.index(max(probabilites_2))

         max_index_2=probabilites_3.index(max(probabilites_3))

         max_index_3=probabilites_4.index(max(probabilites_4))

         max_index_4=probabilites_5.index(max(probabilites_5))

          # Collect the indices in a list.

         self.index_list=[max_index_0,max_index_1,max_index_2,max_index_3,max_index_4]

         # Determine the most common prediction across all models.

         self.max_index=max(set(self.index_list), key=self.index_list.count)

         #Determine number of times each index occurs and storing in list

         self.frequency_list=[self.index_list.count(item) for item in self.index_list] 

         # Evaluate whether the models have converged in to a single prediction

         self.update_convergence()

          # Display the current convergence status on the screen.
         
         text_size = cv2.getTextSize(self.text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]

         x = int((self.frame.shape[1] - text_size[0]) / 2)

         cv2.putText(self.frame, self.text, (x, 30),
         
         cv2.FONT_HERSHEY_SIMPLEX, 1, (208, 224, 64), 2)

         cv2.imshow('frame', self.frame)

         # Wait for the user to press 'q', which signifies that they are ready to proceed with the first prediction when convergence criterion is met
         
         if (cv2.waitKey(1) & 0xFF == ord('q')):
            
            tracking_convergence=True

         # Once convergence is achieved and confirmed by the user, exit the loop.
         if tracking_convergence and self.converged:

            break   

       # Release the video capture object and close the OpenCV window to clean up.
      self.cap.release()
      cv2.destroyAllWindows()

      # Reinitialize the video capture for future predictions or game rounds.
      self.cap = cv2.VideoCapture(0)

      # Return the index of the gesture that was most commonly predicted by the models.
      return self.max_index   
   
   @staticmethod
   def user_aid(func):
      '''
      A decorator used to provide real-time aid to the user during the game.
      It gives visual cues or instructions if the model has not converged on a prediction
      or if additional guidance is needed for the user to position their hand correctly.

      Parameters:
         func (function): The function that this decorator is applied to i.e. a method of RockPaperScissors class
                           that requires showing aid to the user.
      '''
      
      def wrapper(self):
         '''
         The wrapper function that is called instead of the original function. It checks if aid should be displayed
         based on the game's current state and calls the original function with added functionality.

         Parameters:
               self: The instance of the class where the decorated method belongs.
         '''
         
         # Start a timer if it's not already started and if the model hasn't converged yet.
         # This helps in managing when to display the aid text on the screen.
         if self.timer_started is False and self.converged is False:
               self.start_time = clock.time()  # Record the current time to start the timer.
               self.timer_started = True  # Mark the timer as started.

         # Call the original function that this decorator is applied to.
         func(self)

         # Calculate how long it's been since the aid timer was started.
         self.end_time = clock.time()  # Get the current time to calculate the duration.
         duration = self.end_time - self.start_time  # Calculate the duration.

         # Check if enough time has passed without convergence; if so, display the aid text.
         if duration > 1.5 and self.converged is False:
               # Text size and positioning for the aid message.
               text_1_size = cv2.getTextSize("Move your hand around, closer and" , cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
               text_2_size = cv2.getTextSize("further, twist, rotate and turn." , cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]

               # Calculate the position for the aid text to ensure it's centered.
               x_1 = int((self.frame.shape[1] - text_1_size[0]) / 2)
               y_1 = int((self.frame.shape[0] - text_1_size[1]) / 2 - 30)
               x_2 = int((self.frame.shape[1] - text_2_size[0]) / 2)
               y_2 = int((self.frame.shape[0] - text_2_size[1]) / 2 + 30)
               
               # Display the aid messages on the frame.
               cv2.putText(self.frame, "Move your hand around, closer and", (x_1, y_1),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (147, 20, 255), 2)
               cv2.putText(self.frame, "further, twist, rotate and turn.", (x_2, y_2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (147, 20, 255), 2)

               # Show the frame with the aid text to the user.
               cv2.imshow('frame', self.frame)

         # If the model has converged, reset the timer for the next use.
         elif self.converged is True:
               self.start_time = clock.time()  # Resetting the timer.

      # Return the wrapper function to replace the original function.
      return wrapper

   @user_aid
   def update_convergence(self):
      '''
      This method checks if the prediction models have converged to a single prediction. Convergence is determined based
      on the frequency of the most common prediction among the latest set of predictions from all models. If a sufficient
      number of models agree on the same prediction, the method updates the class state to indicate that convergence has
      been achieved.

      Updates the 'converged' attribute to True if convergence criteria are met, otherwise sets it to False. Also updates
      the 'text' attribute to reflect the current convergence status.
      '''
      
      # Check if there's a sufficient consensus among model predictions for convergence.
      #  4 or 5 out of 5 models agreeing on the same gesture is condiered as convergence.
      #Further more self.max_index==3 indicates the classification is a not recognised shape, no convergence is assumed then.
      if (4 in self.frequency_list or 5 in self.frequency_list) and self.max_index!=3:
         # If there is sufficient consensus among the models, mark the prediction as converged.
         self.converged = True
         self.text = "Model Converged"  # Update the text to indicate convergence.
      else:
         # If there is not enough consensus, mark the prediction as not converged.
         self.converged = False
         self.text = "Not Converged"  # Update the text to indicate lack of convergence.


   def get_round_winner(self, user_choice, computer_choice):
      '''
      Determines the winner of the current game round by comparing the user's choice against the computer's choice.
      Updates the game state based on the outcome: increases win counts for either the user or the computer,
      and sets the round outcome message. In the event of a draw, the number of rounds remains unchanged,
      allowing for a re-match of the round.

      Parameters:
         user_choice (str): The hand gesture chosen by the user (Rock, Paper, or Scissors).
         computer_choice (str): The hand gesture randomly chosen by the computer (Rock, Paper, or Scissors).
      '''
      
      # Check for a tie between user and computer choices.
      if user_choice == computer_choice:
         # Update the round outcome message to indicate a tie. No winner, so the number of rounds remains the same.
         self.round_outcome = f"It is a tie! Computers choice is: {computer_choice}."

      # Determine if the user wins based on the game's rules.
      elif (user_choice == "Rock" and computer_choice == "Scissors") or \
            (user_choice == "Paper" and computer_choice == "Rock") or \
            (user_choice == "Scissors" and computer_choice == "Paper"):
         # User wins this round. Update the round outcome message and increment the user's win count.
         self.round_outcome = f"You won this round! Computers choice is: {computer_choice}."
         self.user_wins += 1
         # Decrement the number of remaining rounds as this round is concluded with a definitive outcome.
         self.no_of_rounds -= 1

      # If it's not a tie and the user didn't win, then the computer wins.
      else:
         # Computer wins this round. Update the round outcome message and increment the computer's win count.
         self.round_outcome = f"You lost this round! Computers choice is: {computer_choice}."
         self.computer_wins += 1
         # Decrement the number of remaining rounds as this round is concluded with a definitive outcome.
         self.no_of_rounds -= 1

   def play_a_round(self):
    '''
    Executes the logic for playing a single round of Rock, Paper, Scissors.
    This method captures the user's hand gesture through the webcam, generates a random choice for the computer,
    and then determines the winner of the round. It updates the game state based on the round's outcome.

    The method leverages the `get_prediction` function to interpret the user's hand gesture and uses
    random selection for the computer's move. The outcome is determined by comparing these choices,
    and the game state is updated accordingly.
    '''

    # Generate a random choice for the computer among Rock (0), Paper (1), and Scissors (2).
    computer_input = rand.randint(0, 2) 

    # Obtain the user's gesture as a prediction from the webcam input.
    user_input = self.get_prediction()

    # If the prediction is 'Nothing' (3), prompt the user to try again.
    # This loop ensures a valid hand gesture is captured before proceeding.
    while user_input == 3:
        print('Cannot classify this camera image into rock, paper, or scissors. Please try again ensuring that the correct shape takes most of the camera display side length.')
        user_input = self.get_prediction()  # Attempt to capture and predict the hand gesture again.

    # Convert numerical inputs to their corresponding string representations for both user and computer choices.
    computer_choice = self.number_key_mapping_dict[computer_input]  # Convert computer's numeric choice to string.
    user_choice = self.number_key_mapping_dict[user_input]  # Convert user's numeric prediction to string.

    # Determine the winner of the round based on the choices made by the user and the computer.
    self.get_round_winner(user_choice, computer_choice)

   def check_game_status(self): 
     
     '''
    Evaluates the current status of the game by checking the number of rounds remaining and comparing the win counts
    of the user and the computer. If all rounds have been played, it determines the overall winner of the game based
    on the win counts, updates the game outcome message, and then displays the final game results.

    This method is crucial for transitioning the game from ongoing rounds to concluding the game, providing closure
    on the match by declaring a final winner or indicating a tie.

    Returns:
       no_of_rounds (int): The number of rounds left to play. A value of -1 indicates the game has ended.
        
     '''
     # Check if there are no more rounds left to play.
     if(self.no_of_rounds==0):
        
        # The game has ended. Determine the winner or if there's a tie.
        print("Game has ended.")
        
        # The user has won more rounds than the computer.
        if(self.user_wins>self.computer_wins):

           print(f"You win the game, {self.user_wins} to {self.computer_wins}.")

           self.game_outcome=f"You win the game!"

           self.no_of_rounds-=1

         # The computer has won more rounds than the user.
        elif(self.user_wins<self.computer_wins):

           print(f"You lose the game, {self.user_wins} to {self.computer_wins}.")

           self.game_outcome=f"You lose the game!"

           self.no_of_rounds-=1

        else:
            # Both the user and the computer have won the same number of rounds.
           print(f"The game ends in a tie, both you and computer won {self.user_wins} game rounds.")

           self.game_outcome=f"The game ends in a tie!"

        self.display_game_end_result()  

     return self.no_of_rounds

  
def play_a_game(no_of_rounds, lighting_condition):
    
    '''
    Start and manage a game of Rock, Paper, Scissors with the specified number of rounds.

    Parameters:
        no_of_rounds (int): The total number of rounds to be played.

        lighting_condition (string): Room lighting condition.
    '''
    #intialising game instance

    game = RockPaperScissors(no_of_rounds, lighting_condition)

    while game.no_of_rounds>0:
       
       game.play_a_round()

       game.display_game_rounds()

       game.check_game_status()

@click.command()

@click.option('--rounds', '-r', type=int, help='Number of game rounds.')

@click.option('--lighting', '-l', type=click.Choice(['daylight', 'artificial', 'unspecified'], case_sensitive=False), default='unspecified', help='Specify lighting condition: daylight, artificial, or unspecified.')

def main(rounds,lighting):
    
    '''
    The main entry point for the command-line interface.

    Parameters:
        rounds (int): The number of rounds to play, passed as a command-line argument.

        lighting_conditions (string): The lighting conditions in the room.
    '''

    if rounds is None:
        
        rounds = 3  # default value if not provided
        
    play_a_game(rounds,lighting)

if __name__ == "__main__":
    
    main()



