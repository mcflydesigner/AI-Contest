import cv2
import os
import numpy as np
import random as rand
from datetime import datetime
import argparse

""" 
    Global constants. 
    CAN BE CHANGED.
"""
# Please, note: path to the input image relatively to main.py file
# is provided as the 1st argument to the script.

# The size of 1 population(must be greater or equal to 4)
POPULATION_SIZE = 10
# The algorithm terminates either it reaches maximum number of iterations,
# either it reaches stopping criteria during the execution. I limited this
# algorithm by MAX_NUM_OF_ITERATIONS to prevent too long executions.
MAX_NUM_OF_ITERATIONS = 10000
STOPPING_CRITERIA = 5000000
# Default output filename the output file will be saved in the same dir
# with the script
OUTPUT_FILENAME = 'output_' + str(int(datetime.now().timestamp())) + '.jpg'
"""
    End of global constants.
"""


class ExtendedImage:
    """
        Class to work with numpy.ndarray with some additional(helping) methods
    """

    def __init__(self, img):
        assert isinstance(img, np.ndarray), 'Provided image is not instance of `np.ndarray` class'
        self.img = img

    def view(self, name_of_window="Image") -> None:
        """ Open a separate window to show the image """
        cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
        cv2.imshow(name_of_window, self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def get_color_in_region(self, start, end):
        """
            Method which finds average color in the given region
            (start_x, start_y), (end_x, end_y).
        """
        # Input format: (start_x, start_y), (end_x, end_y)
        start_x, start_y = start
        end_x, end_y = end

        # x and y are flipped
        crop_img = self.img[start_x:(end_x + 1), start_y:(end_y + 1)]
        channels = cv2.mean(crop_img)

        # Return BGR
        return channels[0], channels[1], channels[2]

    def get_width(self):
        """ Method to get the width of the image """
        width = np.size(self.img, 0)
        return width

    def get_height(self):
        """ Method to get the height of the image """
        height = np.size(self.img, 1)
        return height

    @staticmethod
    def create_empty_image(width=512, height=512):
        """ Method to create an empty image filled using black color """
        blank_img = np.zeros((width, height, 3), np.uint8)
        # Return instance of the class
        return ExtendedImage(blank_img)


class GeneticAlgorithm:
    """
        Main class of the genetic algorithm.

        Chromosome - image.
        Population consists of chromosomes.
        Fitness function - squared sum of difference between original one and a given image.
        Mutation adds 1 shape on the image.
        Crossover unites the left half of the 1st image with the half of the 2nd image.

        Please, note: extended image means instance of the type ExtendedImage.
    """

    def __init__(self, orig_img, pop_size, stop_crit, max_iter_num):
        # Basic checks for correctness of the input values
        assert pop_size > 4, 'The size of the population must be >= 4'
        assert stop_crit > 0, 'The value of stopping criteria must be > 0'
        assert max_iter_num > 0, 'The number of iteration must be > 0'

        super().__init__()
        self._orig_img = ExtendedImage(orig_img)
        self._pop_size = pop_size
        self._stop_crit = stop_crit
        self._max_iter_num = max_iter_num

    def _generate_population(self) -> None:
        """
            Protected method to generate initial population. Len of initial population is pop_size.
            After the generation, the population is saved inside the object(self._population).
        """
        self._population = list()
        blank_img_ext = ExtendedImage.create_empty_image(width=self._orig_img.get_width(),
                                                         height=self._orig_img.get_height())
        initial_fitval = self._fit_test(blank_img_ext)

        for i in range(self._pop_size):
            # Each chromosome is an empty black image
            blank_img_ext = ExtendedImage.create_empty_image(width=self._orig_img.get_width(),
                                                             height=self._orig_img.get_height())
            # Form of 1 element of the population: (member, fitness value)
            self._population.append((blank_img_ext, initial_fitval))

    def _fit_test(self, img_ext) -> int:
        """
            Protected method evaluates the goodness of the chromosome calculating the
            squared sum of difference between the original one and a given extended image(img_ext)
        """
        sum_squared_difference = np.sum(np.square(self._orig_img.img - img_ext.img))

        return sum_squared_difference

    def _mutate(self, img_ext) -> ExtendedImage:
        """
            Protected method takes an extended image(img_ext) and adds one rectangle
            on the image using alpha channel(some opacity) returning mutated member of the population.
            The color of the new figure is calculated using the original image: mean(average)
            color of the cropped part of the original img(it is used due to optimization of the genetic algorithm).
            In addition, the size of the the rectangle varies from 5px up to 9px for a good approximation
            of mean color.
        """
        width = img_ext.get_width()
        height = img_ext.get_height()

        # randomly choose y coord. of stating point for the rectangle,
        # but we are limited by the height of the image
        start_y = rand.randint(0, height - 1)
        # randomly choose y coord. of the final point for the rectangle
        end_y = min(start_y + rand.randint(4, 8), height - 1)

        # randomly choose x coord. of stating point for the rectangle,
        # but we are limited by the width of the image
        start_x = rand.randint(0, width - 1)
        # randomly choose x coord. of the final point for the rectangle
        end_x = min(start_x + rand.randint(4, 8), width - 1)

        # find the mean color of the cropped area on the original picture
        color = self._orig_img.get_color_in_region((start_x, start_y), (end_x, end_y))

        # Without overlay
        # cv2.rectangle(copy_member.img, (start_y, start_x), (end_y, end_x), color, -1)

        # With overlay
        # Copy the original image and add the new rectangle on it
        overlay = img_ext.img.copy()
        cv2.rectangle(overlay, (start_y, start_x), (end_y, end_x), color, -1)
        # Randomly choose opacity of the new rectangle
        alpha = rand.uniform(0, 1)
        mutated_member = cv2.addWeighted(overlay, alpha, img_ext.img, 1 - alpha, 0)
        return ExtendedImage(mutated_member)

    def _crossing_over(self, img_ext_1, img_ext_2) -> ExtendedImage:
        """
            Protected method is the implementation of crossing over(Chromosomal crossover).
            The crossover unites the left half of the 1st image with the half of the 2nd image
            returning new instance member of the population.
        """
        # Copy first extended image
        new_member = img_ext_1.img.copy()
        height = img_ext_2.get_height()

        # Add the right half of the 2nd image to copy of the 1st image
        new_member[0:, (height // 2):, :3] = img_ext_2.img[0:, (height // 2):, :3]
        return ExtendedImage(new_member)

    def _selection(self) -> None:
        """
            Protected method to generate the next population based on the
            previous one.
        """
        # The size of the new population must be the same as the prev. one
        max_size_of_pop = self._pop_size

        # Copy 50% of best chromosomes to the next generation
        num_of_pop_to_next_gen = round(self._pop_size / 2)
        max_size_of_pop -= num_of_pop_to_next_gen
        self._population = self._population[0:num_of_pop_to_next_gen]

        # Mutate 25% of the prev. population and add to the next generation
        num_of_mutated_to_next_gen = round(max_size_of_pop / 2)
        max_size_of_pop -= num_of_mutated_to_next_gen
        for i in range(num_of_mutated_to_next_gen):
            # Mutate one member from the prev. generation
            img, _ = self._population[i]
            new_mutated_member = self._mutate(img)

            # Apply more mutation to one chromosome(from 0 to 100)
            for i in range(rand.randint(0, 100)):
                new_mutated_member = self._mutate(new_mutated_member)

            # Evaluate the goodness of obtained chromosome
            fitval = self._fit_test(new_mutated_member)
            # Add the mutated chromosome to the next generation
            self._population.append((new_mutated_member, fitval))

        # For remaining 25% of the prev. population do crossing overs
        num_of_crossing_overs_to_next_gen = max_size_of_pop
        max_size_of_pop -= num_of_crossing_overs_to_next_gen

        for i in range(num_of_crossing_overs_to_next_gen):
            # Choose 2 chromosomes, then do one crossing over
            img_ext_1, _ = self._population[i]
            img_ext_2, _ = self._population[rand.randint(0, num_of_pop_to_next_gen)]

            new_mutated_member = self._crossing_over(img_ext_1, img_ext_2)
            # Evaluate the goodness of obtained chromosome
            fitval = self._fit_test(new_mutated_member)
            # Add the derived chromosome to the next generation.
            # Form of 1 element of the population: (member, fitness value)
            self._population.append((new_mutated_member, fitval))

        # Sort the new generation in increasing order based on the fitness value of each chromosome
        self._population.sort(key=lambda x: x[1])
        print(f'Best chromosome fit value: {self._population[0][1]}')

    def run(self) -> (ExtendedImage, int):
        """
            Public method to run the genetic algorithm.
            Returns the tuple of 2 elements: (ExtendedImage, fitness value)

            This method returns the best obtained picture which was obtained
            during all iterations. The algorithm terminates either it reaches
            STOPPING_CRITERIA or MAX_NUM_OF_ITERATIONS.
        """
        # Generate an initial population
        print('Generating the first population...')
        self._generate_population()
        print('The first population generated! Starting the algorithm...')

        # Start the algorithm itself to obtain the solution for the problem
        for i in range(self._max_iter_num):
            # Also calculate execution time of 1 iteration
            start = datetime.now()
            # Run selection to generate a new population based on the prev. one
            self._selection()
            end = datetime.now()
            print(f'Iteration {i} finished; time = {(end - start).total_seconds()}')

            # If STOPPING_CRITERIA is reached, terminate the algorithm
            # (the best chromosome in the new generation is good enough)
            if self._population[0][1] <= self._stop_crit:
                print('Stopping criteria reached. Terminate.')
                break

        print('Algorithm finished!')
        # Return obtained result in the form of tuple (picture, fitness value)
        return self._population[0]




""" 
    Functions to validate the input arguments provided by users.
"""
def check_input_file(value):
    """
        Function to check that a given file(value) exists
    """
    if not os.path.exists(value):
        raise argparse.ArgumentTypeError(f'Input file `{value}` does not exist')

    return value


def check_positive(value):
    """
        Function to check that the int input value is positive
    """
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f'must be positive value, but got {value}')
    return ivalue


def check_pop_size(value):
    """
        Function to check that the int input value is >= 4
    """
    ivalue = int(value)
    if ivalue < 4:
        raise argparse.ArgumentTypeError(f'The size of the population must be >= 4, '
                                         f'but got {value}')
    return ivalue


""" Main function """
if __name__ == '__main__':
    # Set up argument parser to validate the input of the user
    parser = argparse.ArgumentParser()
    # Only 1 required argument: input filename
    parser.add_argument('input_filename', help='Input filename', type=check_input_file)
    # Optional arguments
    parser.add_argument('-p', help='Population size', type=check_pop_size, nargs='?',
                        default=POPULATION_SIZE)
    parser.add_argument('-s', help='Stopping criteria', type=check_positive, nargs='?',
                        default=STOPPING_CRITERIA)
    parser.add_argument('-m', help='Max iteration number', type=check_positive, nargs='?',
                        default=MAX_NUM_OF_ITERATIONS)
    parser.add_argument('-o', help='Output filename', type=str, nargs='?', default=OUTPUT_FILENAME)
    args = parser.parse_args()

    # Using BGR, and NOT RGB while working with img
    image = cv2.imread(args.input_filename)
    # Set up the genetic algorithm
    g = GeneticAlgorithm(orig_img=image,
                         pop_size=args.p,
                         stop_crit=args.s,
                         max_iter_num=args.m)
    # Run the generic algorithm
    result_img_ext, _ = g.run()
    # Show the obtained result
    result_img_ext.view()

    # Save output file(if default: using timestamp for uniqueness of the name)
    filename = args.o
    cv2.imwrite(filename, result_img_ext.img)
