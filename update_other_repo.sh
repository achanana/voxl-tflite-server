#!/bin/bash
################################################################################
# Copyright (c) 2022 ModalAI, Inc. All rights reserved.
#
# This is a script to copy new versions of some of the common files in this
# template into other repositories. Specifically it will copy the following
# files into another repository
#
# clean.sh
# deploy_to_voxl.sh
# make_package.sh
# .gitignore
# LICENSE
#
# When the template updates, this saves time in updating other projects to match
################################################################################

YES=false
DIRS=""

print_usage () {
	echo -e ""
	echo -e "Updates common template files in one or more other repositories."
	echo -e "./update_other_repo.sh ../voxl-mavlink-server ../libmodal_pipe"
	echo -e ""
	echo -e "To silence the questions and just copy everything anyway,"
	echo -e "use the -Y argument"
	echo -e "./update_other_repo.sh -Y ../voxl-mavlink-server ../libmodal_pipe"
	echo -e ""
	exit 0
}

update_file () {

	if $YES; then
		if ! [ -f "$1/$2" ]; then
			echo "$1 is missing $2, copying"
			cp $2 $1/$2
		elif ! cmp -s $1/$2 $2; then
			echo "$1/$2 differs from template file, copying"
			cp $2 $1/$2
		fi
	else
		if ! [ -f "$1/$2" ]; then
			echo "$1 is missing $2"
			read -p "Press enter to copy template file"
			cp $2 $1/$2
		elif ! cmp -s $1/$2 $2; then
			echo "$1/$2 differs from template file"
			read -p "Press enter to copy template file"
			cp $2 $1/$2
		else
			echo "$2 matches, skipping"
		fi
	fi
}

update_dir () {

	echo "scanning $1"

	if [ -f "$1/install_on_voxl.sh" ]; then
		echo "install_on_voxl.sh has been replaced with deploy_to_voxl.sh"
		read -p "Press enter to remove"
		rm -f "$1/install_on_voxl.sh"
	fi

	update_file $1 "clean.sh"
	update_file $1 "deploy_to_voxl.sh"
	update_file $1 "make_package.sh"
	update_file $1 ".gitignore"
	update_file $1 "LICENSE"

}


# parse arguments
for i in $@; do
	if [ "$i" == "-Y" ] || [ "$i" == "-y" ]; then
		echo "enabling yes for all"
		YES=true;
	else
		NEW_PATH=$(readlink -f $i)
		DIRS="$DIRS $NEW_PATH"
	fi
done


if [[ "$DIRS" == "" ]]; then
	echo "No directory given"
	print_usage
fi

# do each dir
echo "about to update the following directories"
for i in $DIRS; do
	echo $i
done

if ! $YES; then
	read -p "Press enter to continue"
fi

for i in $DIRS; do
	update_dir $i
done


echo "DONE"
