#!/bin/bash
################################################################################
# Copyright 2021 ModalAI Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# 4. The Software is used solely in conjunction with devices provided by
#    ModalAI Inc.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
################################################################################

NAME="voxl-tflite-server"
SERVICE_FILE="${NAME}.service"
CONFIG_FILE="/etc/modalai/${NAME}.conf"
USER=$(whoami)


print_usage () {
	echo ""
	echo "Start wizard with prompts:"
	echo "voxl-configure-tflite-server"
	echo ""
	echo "Shortcut configuration arguments for scripted setup."
	echo "factory_enable will reset the config file to factory defaults"
	echo "before enabling the service."
	echo ""
	echo "voxl-configure-tflite-server disable"
	echo "voxl-configure-tflite-server factory_enable"
	echo "voxl-configure-tflite-server enable"
	echo ""
	echo "show this help message:"
	echo "voxl-configure-tflite-server help"
	echo ""
	exit 0
}


## set most parameters which don't have quotes in json
set_param () {
	if [ "$#" != "2" ]; then
		echo "set_param expected 2 args"
		exit 1
	fi

	# remove quotes if they exist
	var=$1
	var="${var%\"}"
	var="${var#\"}"
	val=$2
	val="${val%\"}"
	val="${val#\"}"

	sed -E -i "/\"$var\":/c\	\"$var\":	$val," ${CONFIG_FILE}
}

## set string parameters which need quotes in json
set_param_string () {
	if [ "$#" != "2" ]; then
		echo "set_param_string expected 2 args"
		exit 1
	fi
	var=$1
	var="${var%\"}"
	var="${var#\"}"
	sed -E -i "/\"$var\":/c\	\"$var\":	\"$2\"," ${CONFIG_FILE}
}


disable_service_and_exit () {
	echo "disabling ${NAME} systemd service"
	systemctl disable ${SERVICE_FILE}
	echo "stopping ${NAME} systemd service"
	systemctl stop ${SERVICE_FILE}
	echo "Done configuring ${NAME}"
	exit 0
}

enable_service_and_exit () {
	echo "enabling  ${NAME} systemd service"
	systemctl enable  ${SERVICE_FILE}
	echo "starting  ${NAME} systemd service"
	systemctl restart  ${SERVICE_FILE}
	echo "Done configuring ${NAME}"
	exit 0
}

reset_config_file_to_default () {
	echo "wiping old config file"
	rm -rf ${CONFIG_FILE}
	${NAME} -c
}


################################################################################
## actual start of execution, handle optional arguments first
################################################################################

## sanity checks
if [ "${USER}" != "root" ]; then
	echo "Please run this script as root"
	exit 1
fi


## convert argument to lower case for robustness
arg=$(echo "$1" | tr '[:upper:]' '[:lower:]')

## parse arguments
case ${arg} in
	"")
		echo "Starting Wizard"
		;;
	"h"|"-h"|"help"|"--help")
		print_usage
		exit 0
		;;
	"disable")
		disable_service_and_exit
		;;
	"factory_enable")
		reset_config_file_to_default
		enable_service_and_exit
		;;
	"enable")
		enable_service_and_exit
		;;
	*)
		echo "invalid option"
		print_usage
		exit 1
esac


################################################################################
## no optional arguments, start config wizard prompts
################################################################################


echo " "
echo "Do you want to reset the config file to factory defaults?"
select opt in "yes" "no"; do
case $opt in
yes )
	reset_config_file_to_default
	break;;
no )
	echo "loading and updating config file with ${NAME} -c"
	${NAME} -c
	break;;
*)
	echo "invalid option"
	esac
done


echo " "
echo "do you want to enable or disable ${NAME}"
select opt in "enable" "disable"; do
case $opt in
enable )
	#enable_service_and_exit
	break;;
disable )
	disable_service_and_exit
	break;;
*)
	echo "invalid option"
	esac
done

echo ""
echo -e "do you want to run the tflite server with:\n (1) Mobilenet + Hires camera\n (2) Mobilenet + Tracking Camera\n (3) Pydnet + Hires camera"
select opt in "1" "2" "3"; do
case $opt in
1 )
	set_param_string("model" "/usr/bin/dnn/mobilenet_v1_ssd_coco_labels.tflite")
    set_param_string("input_pipe" "/run/mpa/hires_preview/")
	break;;
2 )
	set_param_string("model" "/usr/bin/dnn/mobilenet_v1_ssd_coco_labels.tflite")
    set_param_string("input_pipe" "/run/mpa/tracking/")
	break;;
3 )
    set_param_string("model" "/usr/bin/dnn/tflite_pydnet.tflite")
    set_param_string("input_pipe" "/run/mpa/hires_preview/")
	break;;
*)
	echo "invalid option"
	esac
done

enable_service_and_exit
# all done!
