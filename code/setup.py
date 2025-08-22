from setuptools import setup, find_packages

setup(
    name='R3DR',
    version='0.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'R3DR-init=R3DR.script.nerf_initialization:main',  # <--- Define entry point 
            'R3DR-correspondence=R3DR.script.nerf_correspondence:main',  # <--- Define entry point
            'R3DR-render=R3DR.script.render:main',
            'R3DR-process-t2=R3DR.script.nerf_process_t2:main'
        ],
    },
    install_requires=[
        # Dependency package list        
    ],
)
