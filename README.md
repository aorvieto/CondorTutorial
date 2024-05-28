# CondorTutorial
A tutorial for new PhD students and Interns at MPI. No efficiency trick, just the minimal setup to run a sweep!

1) Login into the cluster from VScode, open up a terminal.

2) Set up github on cluster (to download this repo). Run these instructions in parallel:
    1) `git config --global user.name "Student Name"`
    2) `git config --global user.email "your_email@mail.com"`
    3) `ssh-keygen -t ed25519 -C "your_email@mail.com"`
    4) `eval "$(ssh-agent -s)"`
    5) `ssh-add ~/.ssh/id_ed25519`
    6) `cat ~/.ssh/id_ed25519.pub`
    7) Go to `https://github.com/settings/keys` and enter the string you got as output, save the SSH key with any name.
