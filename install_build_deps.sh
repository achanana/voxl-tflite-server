################################################################################
# * Copyright 2020 ModalAI Inc.
# *
# * Redistribution and use in source and binary forms, with or without
# * modification, are permitted provided that the following conditions are met:
# *
# * 1. Redistributions of source code must retain the above copyright notice,
# *    this list of conditions and the following disclaimer.
# *
# * 2. Redistributions in binary form must reproduce the above copyright notice,
# *    this list of conditions and the following disclaimer in the documentation
# *    and/or other materials provided with the distribution.
# *
# * 3. Neither the name of the copyright holder nor the names of its contributors
# *    may be used to endorse or promote products derived from this software
# *    without specific prior written permission.
# *
# * 4. The Software is used solely in conjunction with devices provided by
# *    ModalAI Inc.
# *
# * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OFe
# * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# * POSSIBILITY OF SUCH DAMAGE.
################################################################################

#!/bin/bash
################################################################################
# Copyright (c) 2020 ModalAI, Inc. All rights reserved.
################################################################################

cd modalai
tar xvf build-dependencies.tar.gz > /dev/null
cp -R ./build-dependencies/usr/include/* /usr/include/ 2>/dev/null
mkdir /usr/lib64/ 2>/dev/null
cp -R ./build-dependencies/usr/lib64/* /usr/lib64/ 2>/dev/null
cp -R ./build-dependencies/usr/share/* /usr/share/ 2>/dev/null
rm -rf build-dependencies 2>/dev/null
echo ""
echo "Done installing build dependencies"
echo ""

exit 0