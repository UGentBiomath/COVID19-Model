
## Git workflow

This step-by-step git approach is a your _safe haven_ on working with git. When you stick to these guidelines,
you will be succesfully using the 'Github workflow' we apply for the COVID 19 model.
The `master` branch is holy to us and should be on all time a working, up-to-date representation of the functionalities!

This workflow flow assumes you have [forked](https://help.github.com/en/github/getting-started-with-github/fork-a-repo) the COVID repository of Biomath (called `upstream` by default) to your own Github account (called `origin` by default) and cloned your own version to your computer.

### Before I start working

* **STEP 1**: Update the master branch on my PC to make sure it is aligned with the remote master (`upstream`):

        git checkout master
        git pull upstream master


* **STEP 2**: Choose your option:
    * **OPTION 2A**: I already have a branch I want to continue working on:
    Switch to existing topic branch:

            git checkout name_existing_branch
            git fetch upstream
            git merge upstream/master

        (_Note:_ Normally you are the only working on your branch. In the rare case someone updated your branch, update your branch with the remote adjustments: `git merge upstream/name_existing_branch`)

    * **OPTION 2B**:  I'll make a new branch to work with: Create a new topic branch from master(!):


            git checkout master
            git checkout -b name_new_branch


### While editing

*  **STEP 3.x**: adapt in tex, code,... (multiple times)
    * **New files added**


            git add .

    * **Adaptation**

            git commit -am "clear and understandable message about edits"

### Edits on branch ready
* **STEP 4**: Pull request to add your changes to the current master. Choose your option:
    * **OPTION 2A CHOSEN**:

            git push origin name_existing_branch

    * **OPTION 2B CHOSEN**:

            git push origin name_new_branch

* **STEP 5**: Code review!

    Go to your repo online and click the *New pull request* block or click the link in the message in the terminal. This will guide you to the Biomath github repo to make the pull request. You and collaborators can make comments about the edits and review the code.

    If everything is ok, click the  *Merge pull request*, followed by *confirm merge*. Delete the online branch, since obsolete.

    You're work is now tracked and added to master! Congratulations.

    If the code can't be merged automatically (provided by a message online), go to **STEP 7**.

* **STEP 6**: Update local master and remove local merged branch

        git checkout master
        git pull upstream master
        git branch -d name_merged_branch

### Pull request can not be merged
* **STEP 7**: master has changed and there are conflicts: update your working branch with rebase


        git checkout name_existing_branch
        git fetch origin
        git merge upstream/master
        # fix conflicts local
        git add file_with_conflict
        git commit file_with_conflict
        git push origin name_existing_branch


