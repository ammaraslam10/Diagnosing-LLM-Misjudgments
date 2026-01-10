import os
from radon.complexity import cc_rank, cc_visit
from radon.metrics import h_visit, mi_visit
from radon.raw import analyze
import autopep8
import subprocess
import pandas as pd
import re
import csv
from pathlib import PurePosixPath
from io import StringIO

basePath = './'
folderName = 'extracted_codes'
reportFolderName = 'CSV_Reports'
reportFileCommonName = ''


def runBandit(folderName):
  completePath = os.path.join(basePath, folderName)
  reportPath = os.path.join(basePath, reportFolderName)
  banditReport = os.path.join(reportPath, reportFileCommonName + 'bandit.csv')

  # Ensure the report directory exists
  os.makedirs(reportPath, exist_ok=True)

  # Define the command
  command = ['bandit', '-r', completePath, '-f', 'csv', '-vv', '-o', banditReport]

  # Run the command
  try:
      result = subprocess.run(command, check=False, text=True, capture_output=True)
      print("Bandit Command executed successfully.")
  except subprocess.CalledProcessError as e:
      print("Error running command:", e)

# !bandit -r sample_data/Original -f csv -o sample_data/CSVReports/banditResult.csv

"""# **Step 8: Running Pylint**"""

def runPylint(folderName):
  completePath = os.path.join(basePath, folderName)
  completePath = os.path.join(completePath,'*.py')
  reportPath = os.path.join(basePath, reportFolderName)
  jsonReport = os.path.join(reportPath,reportFileCommonName + 'pylint.json')
  pylintReport = os.path.join(reportPath, reportFileCommonName + 'pylint.csv')

  command = [
    'pylint',
    completePath,
    '--output-format=json:'+ jsonReport
  ]

  # Run the command
  try:
      result = subprocess.run(command, check=False, text=True, capture_output=True)
      print("Pylint Command executed successfully.")
  except subprocess.CalledProcessError as e:
      print("An error occurred while running the command.")
      print("Error message:", e.stderr)
      print("Return code:", e.returncode)
      print("Output:", e.output)
  # Load the JSON file
  df = pd.read_json(jsonReport)

  columns = ['module'] + [col for col in df.columns if col != 'module']
  df = df[columns]

  # Convert the DataFrame to a CSV file
  df.to_csv(pylintReport, index=False)

# !pylint sample_data/Original/*.py --output-format=json:sample_data/CSVReports/pylint_output.json

"""# **Step 9: Run Complexipy**"""

# !pip install complexipy
# !complexipy sample_data/Original | tee sample_data/Human_Eval_Dataset_CSV/human_eval_complexipy.txt

def runComplexipy(folderName):
  completePath = os.path.join(basePath, folderName)
  reportPath = os.path.join(basePath, reportFolderName)
  txtReport = os.path.join(reportPath,reportFileCommonName + 'complexipy.txt')
  complexipyReport = os.path.join(reportPath, reportFileCommonName +'complexipy.csv')

  command = "complexipy " + completePath + " | tee " + txtReport

  # Run the command using subprocess with shell=True
  process = subprocess.run(command, shell=True, stderr=subprocess.PIPE, text=True)

  # Check if the process encountered any errors
  if process.returncode == 0:
      print("Complexipy Command executed successfully.")
  else:
      print(f"An error occurred: {process.stderr}")
#   # Initialize an empty list to store rows
#   rows = []

#   # Read the text file and parse it
#   with open(txtReport, 'r') as file:
#       lines = file.readlines()
#       start_collecting = False
#       for line in lines:
#           line = line.strip()
#           if line.startswith('┏'):
#               start_collecting = True
#               continue
#           if line.startswith('┗'):
#               break
#           if start_collecting and line.startswith('│'):
#               # Split line into columns
#               columns = [col.strip() for col in line.split('│')[1:-1]]
#               rows.append(columns)

#   # Write the rows to a CSV file
#   with open(complexipyReport, 'w', newline='') as csvfile:
#       writer = csv.writer(csvfile)
#       # Writing the header
#       writer.writerow(['Path', 'File', 'Function', 'Complexity'])
#       # Writing the data rows
#       writer.writerows(rows)

  # Match any *.py line as a "file line"
  file_re = re.compile(r"^\s*([^\s].*?\.py)\s*$")
  # Match indented lines: <function> <complexity:int> <PASSED|FAILED>
  fn_re = re.compile(r"^\s+([A-Za-z0-9_]+)\s+(\d+)\s+(?:PASSED|FAILED)\s*$")

  rows = []
  current_path = None

  with open(txtReport, 'r') as f:
    text = f.read()

  for raw in text.splitlines():
      line = raw.rstrip()

      m_file = file_re.match(line)
      # Skip header/separator lines
      if m_file and "complexipy" not in line and "─" not in line:
          current_path = m_file.group(1)
          continue

      m_fn = fn_re.match(line)
      if m_fn and current_path:
          function = m_fn.group(1)
          complexity = int(m_fn.group(2))

          p = PurePosixPath(current_path)
          file_name = p.name
          dir_path = "" if str(p.parent) == "." else str(p.parent)

          rows.append([dir_path, file_name, function, complexity])

  # Emit CSV
  with open(complexipyReport, 'w', newline='') as out:
    w = csv.writer(out)
    w.writerow(["Path", "File", "Function", "Complexity"])
    w.writerows(rows)

  print(f"CSV file has been saved to {complexipyReport}")

"""# **Step 10: Run Radon**"""

def get_radon_metrics(file_path):
    # Read the file content
    with open(file_path, 'r', encoding='utf-8') as f:
        code = f.read()

    # Get raw metrics
    raw_metrics = analyze(code)

    # Get cyclomatic complexity metrics
    complexity_metrics = cc_visit(code)

    # Get maintainability index
    maintainability_index = mi_visit(code, True)

    # Get Halstead metrics
    halstead_metrics = h_visit(code)

    return raw_metrics, complexity_metrics, maintainability_index, halstead_metrics

def runRadon(folderName):
  completePath = os.path.join(basePath, folderName)
  reportPath = os.path.join(basePath, reportFolderName)
  radonReport = os.path.join(reportPath, reportFileCommonName +'radon.csv')
  # Prepare the CSV file
  with open(radonReport, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      # Write header
      csvwriter.writerow([
          'File Name', 'LOC', 'LLOC', 'SLOC', 'Comments',
          'Cyclomatic Complexity', 'Maintainability Index',
          'h1', 'h2','h', 'N1', 'N2', 'N',
          'Vocabulary', 'Volume', 'Difficulty',
          'Effort', 'Bugs', 'Time'
      ])

      # Process each .py file in the folder
      for file_name in os.listdir(completePath):
          if file_name.endswith('.py'):
              file_path = os.path.join(completePath, file_name)

              # Get Radon metrics
              try:
                  raw_metrics, complexity_metrics, maintainability_index, halstead_metrics = get_radon_metrics(file_path)
              except Exception as e:
                  print(f"Error processing {file_name}: {e}")
                  continue

              # Calculate total and average complexity
              total_complexity = sum(block.complexity for block in complexity_metrics)


              # Write metrics to CSV
              csvwriter.writerow([
                  file_name,
                  raw_metrics.loc,
                  raw_metrics.lloc,
                  raw_metrics.sloc,
                  raw_metrics.comments,
                  total_complexity,
                  maintainability_index,
                  halstead_metrics.total.h1,  # Number of distinct operators
                  halstead_metrics.total.h2,  # Number of distinct operands
                  halstead_metrics.total.h1 + halstead_metrics.total.h2,
                  halstead_metrics.total.N1,  # Total number of operators
                  halstead_metrics.total.N2,  # Total number of operands
                  halstead_metrics.total.N1 +halstead_metrics.total.N2,
                  halstead_metrics.total.vocabulary,  # Halstead Vocabulary
                  halstead_metrics.total.volume,  # Halstead Volume
                  halstead_metrics.total.difficulty,  # Halstead Difficulty
                  halstead_metrics.total.effort,  # Halstead Effort
                  halstead_metrics.total.bugs,  # Halstead Estimated Bugs
                  halstead_metrics.total.time  # Halstead Time to Implement
              ])

  print(f'All Radon metrics have been saved to {radonReport}')


runBandit(folderName)
runPylint(folderName)
runComplexipy(folderName)
runRadon(folderName)