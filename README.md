# Overview

This repo is a simple example of how Federated Learning can be implemented using [Flower](https://flower.dev).

In this demo we are going to perform federated learning with flower using a server and two clients.

### Setting Up the Environment:
This code is developed and tested on an ubuntu machine with python 3.8 installed and venv for virtual environment management.

1. Clone the repository by pasting the below command in the terminal.
    ```bash
    git clone https://github.com/nclabteam/FL-Demo-With-Flower.git
    ```
2. After the repository is cloned, open the cloned repository in terminal.
    ```bash
    cd FL-Demo-With-Flower/
    ```
3. Ensure pip is installed if not install using
    ```bash
    sudo apt install python3-pip
    ```
4. We are using venv package to create virtual environent you can use any virtual environment libary to proceed. Run the below command in terminal to install venv.
    ```bash
    sudo apt install python3.8-venv
    ```

5. Use python venv to create a virtual environment for our project 

    ```bash
    python3 -m venv venv-flwr-demo
    ```
   Where "venv-flwr-demo" is the name of the virtual environment that we want to create. We can choose any name that we like instead of venv-flwr-demo. After this step a new directory (venv-flwr-demo) will be created.
6. For using virtual environment we need to activate the environment first.
     ```bash
    source venv-flwr-demo/bin/activate
    ```
7. Now we can install the project requirements or dependencies inside virtual environment using terminal as:
    ```bash
    pip3 install -r requirements.txt
    ```
8. Open two more terminal windows in the same directory and again activate the virtual env for those terminals also as shown in step 6.
9. In one terminal run the server.py file
    ```bash
    python3 server.py
    ```
10. Run client1.py in second terminal 
     ```bash
    python3 client1.py
    ```
11. Run client2.py in second terminal 
     ```bash
    python3 client2.py
    ```