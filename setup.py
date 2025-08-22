from setuptools import setup, find_packages

setup(
    name='SCD',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'SCD-init=SCD.script.nerf_initialization:main',  # <--- Define entry point 
            'SCD-correspondence=SCD.script.nerf_correspondence:main',  # <--- Define entry point
            'SCD-render=SCD.script.render:main',
            'SCD-process-t2=SCD.script.nerf_process_t2:main'
        ],
    },
    install_requires=[
        # Dependency package list        
    ],
)
