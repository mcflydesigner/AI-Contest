# AI-Contest
Implementation of a genetic algorithm to produce some art :)

## How to run the project?

## 1. Install *Python 3.9* and *pip* on your machine
A good installation guide is available [here](https://phoenixnap.com/kb/how-to-install-python-3-windows)
## 2. Clone the project on your local machine
```
git clone https://github.com/mcflydesigner/AI-Contest.git
```
## 3. Install all requirements for the project
Move to the main folder of the project.
In your terminal(bash) run the following command
```
pip install -r requirements.txt
```
## 4. Run the program through the terminal
Run the following command in your terminal
```
python main.py PATH_TO_INPUT_IMG
```
instead of PATH_TO_INPUT_IMG provide a relative path to your input picture.

### 4.1 Running the program with specified parameters
The program supports the following **arguments** which **are optional**
`[-p] NUMBER` - size of the population(must be a positive number which is greater or equal to 4, default=10)
`[-s] NUMBER` - stopping criteria(must be a positive number, default=5000000)
`[-m] NUMBER` - maximum number of iterations(must be a positive number, default=10000)
`[-o] NUMBER` - output filename(must be a string, default=`timestamp.now()`)

Example of the command to run the program with optional parameters:
```
python main.py PATH_TO_INPUT_IMG -p 10000 -s 500000 -m 10000 -o output.jpg
```
