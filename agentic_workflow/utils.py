import os
import shutil
import subprocess
import zipfile


def create_deployment_package_no_dependencies(
    lambda_code, project_name, output_zip_name
):
    """
    Create a deployment package without dependencies.
    """
    # Create the project directory
    os.makedirs(project_name, exist_ok=True)

    # Write the lambda code to the lambda_function.py file
    with open(os.path.join(project_name, "lambda_function.py"), "w") as f:
        f.write(lambda_code)

    # Create a .zip file for the deployment package
    with zipfile.ZipFile(output_zip_name, "w") as zipf:
        zipf.write(
            os.path.join(project_name, "lambda_function.py"), "lambda_function.py"
        )

    # Clean up the project directory
    shutil.rmtree(project_name)

    return output_zip_name


def create_deployment_package_with_dependencies(
    lambda_code, project_name, output_zip_name, dependencies
):
    """
    Create a deployment package with dependencies.
    """
    # Create the project directory
    os.makedirs(project_name, exist_ok=True)

    # Write the lambda code to the lambda_function.py file
    with open(os.path.join(project_name, "lambda_function.py"), "w") as f:
        f.write(lambda_code)

    # Install the dependencies to the package directory
    package_dir = os.path.join(project_name, "package")
    os.makedirs(package_dir, exist_ok=True)

    # Turn dependencies into a list
    # dependencies = dependencies.split(",")

    for dependency in dependencies:
        subprocess.run(["pip", "install", "--target", package_dir, dependency])

    # Create a .zip file for the deployment package
    with zipfile.ZipFile(output_zip_name, "w") as zipf:
        # Add the installed dependencies to the .zip file
        for root, _, files in os.walk(package_dir):
            for file in files:
                zipf.write(
                    os.path.join(root, file),
                    os.path.relpath(os.path.join(root, file), package_dir),
                )
        # Add the lambda_function.py file to the .zip file
        zipf.write(
            os.path.join(project_name, "lambda_function.py"), "lambda_function.py"
        )

    # Clean up the project directory
    shutil.rmtree(project_name)

    return output_zip_name
