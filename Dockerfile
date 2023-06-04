FROM continuumio/miniconda3

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc \
       libpq-dev \
       libssl-dev \
       libffi-dev \
       python3-dev

# Create a new Conda environment
RUN conda create -n myenv python=3.8.16

# Activate the Conda environment
SHELL ["conda", "run", "--no-capture-output", "-n", "myenv", "/bin/bash", "-c"]

# Copy requirements.txt and install Python dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
# RUN conda install libpython m2w64-toolchain -c msys
RUN conda update -n base -c defaults conda
RUN conda install numpy holidays cython -c conda-forge
RUN conda install matplotlib scipy pandas -c conda-forge
RUN conda install pystan -c conda-forge
RUN conda install -c anaconda ephem
RUN conda install -c conda-forge fbprophet

# Copy the rest of the application code
COPY . /app

# Set the command to run your Flask application
CMD ["conda", "run", "--no-capture-output", "-n",  "myenv", "gunicorn", "--workers=4", "--bind", "0.0.0.0:5000", "app:app"]
# CMD ["python", "app.py"]

# FROM continuumio/miniconda3:latest

# # Upgrade conda
# RUN conda update -n base -c defaults conda

# # Create a new conda environment
# RUN conda create -n myenv python=3.8.16

# # Activate the conda environment
# # SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
# # Make RUN commands use the new environment:
# SHELL ["conda", "run", "--no-capture-output", "-n", "myenv", "/bin/bash", "-c"]

# COPY requirements.txt /
# # RUN conda activate myenv/
# # Install required packages
# # RUN pip install requirements.txt 
# # RUN pip install --file /requirements.txt -y  
# RUN pip install -r /requirements.txt
# RUN conda install -c conda-forge fbprophet

# # Use '-y' flag to automatically answer 'yes' to prompts
# RUN pip install fbprophet

# # Set the working directory
# WORKDIR /app

# # Copy the application code
# COPY . /app

# # Expose the port
# EXPOSE 5000

# # Specify the command to run the application
# CMD ["conda", "run", "--no-capture-output", "-n",  "myenv", "gunicorn", "--workers=4", "--bind", "0.0.0.0:5000", "app:app"]

# # FROM continuumio/miniconda3:latest
# # # Upgrade conda
# # RUN conda update -n base -c defaults conda

# # # Create a new conda environment
# # RUN conda create -n myenv python=3.8.16

# # # Activate the conda environment
# # SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

# # # Install required packages
# # COPY requirements.txt /
# # # RUN  conda install --file /requirements.txt
# # RUN pip install -r /requirements.txt
# # RUN conda install -c conda-forge fbprophet

# # # Set the working directory
# # WORKDIR /app

# # # Copy the application code
# # COPY . /app

# # # Expose the port
# # EXPOSE 5000

# # # Specify the command to run the application
# # CMD ["conda", "run", "-n", "myenv", "gunicorn", "--workers=4", "--bind", "0.0.0.0:5000", "app:app"]


























# # FROM python:3.8.16

# # # Upgrade pip and setuptools
# # # RUN pip install --upgrade pip setuptools

# # COPY requirements.txt /
# # RUN pip install -r /requirements.txt
# # COPY . /app
# # WORKDIR /app

# # EXPOSE 5000

# # # Specify the command to run your application
# # CMD gunicorn --workers=4 --bind 0.0.0.0:5000 app:app

# # # FROM python:3.8.16

# # # Upgrade pip
# # # RUN pip install --upgrade pip
# # # RUN pip install --upgrade pip
# # # RUN pip install --upgrade setuptools

# # COPY requirements.txt /
# # RUN pip install -r /requirements.txt
# # COPY . /app
# # WORKDIR /app
# # RUN pip install -r requirements.txt,
# #  /usr/local/bin/python -m pip install --upgrade pip,
# #  /usr/local/bin/python -m pip install --upgrade setuptools
# # EXPOSE 5000

# # # Specify the command to run your application
# # CMD gunicorn --workers=4 --bind 0.0.0.0:5000 app:app

# # # # Install virtualenv
# # # RUN pip install virtualenv

# # # # Set working directory
# # # WORKDIR /app

# # # # Create and activate virtual environment
# # # RUN python -m virtualenv venv
# # # ENV PATH="/app/venv/bin:$PATH"

# # # Copy requirements file and install dependencies
# # # COPY requirements.txt .
# # # RUN pip install -r requirements.txt

# # # # Copy the rest of your application files
# # # COPY . .

# # # Expose port
# # # EXPOSE 5000

# # # # Specify the command to run your application
# # # CMD gunicorn --workers=4 --bind 0.0.0.0:5000 app:app

# # # FROM python:3.8.16
# # # RUN /usr/local/bin/python -m pip install --upgrade pip
# # # COPY requirements.txt /
# # # RUN pip install -r /requirements.txt
# # # COPY . /app
# # # WORKDIR /app
# # # RUN pip install -r requirements.txt
# # # EXPOSE 5000
# # # # CMD python app.py
# # # CMD gunicorn --workers=4 --bind 0.0.0.0:5000 app:app   