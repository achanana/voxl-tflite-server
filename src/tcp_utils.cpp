/*
 * Copyright (c) 2015 Qualcomm Technologies, Inc.  All Rights Reserved.
 * Qualcomm Technologies Proprietary and Confidential.
 */

#include <fcntl.h>
#include "tcp_utils.hpp"


int TcpServer::create_socket (const char *ip_addr, int port_num)
{
    sock_ = socket(AF_INET, SOCK_STREAM, 0);

    if (sock_ < 0) 
    {
        fprintf(stderr, "\nCould not create socket");
        return -1;
    }

    fprintf(stderr, "\nCreated socket");

    memset((void *)&serv_addr_, 0, sizeof(serv_addr_));

    serv_addr_.sin_family      = AF_INET;
    serv_addr_.sin_addr.s_addr = inet_addr(ip_addr);
    serv_addr_.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr_.sin_port        = htons(port_num);
    serv_addr_size_            = sizeof(serv_addr_);

    // int flags = fcntl(sock_, F_GETFL);
    // fcntl(sock_, F_SETFL, flags | O_NONBLOCK);

    return 0;
}

int TcpServer::bind_socket (void)
{
    int sockoptval = 1;

    if (setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &sockoptval, sizeof(int)) < 0)
    {
        fprintf(stderr, "\nError with setsockopt");
        return -1;
    }

    int res = bind(sock_, (struct sockaddr *)&serv_addr_, sizeof(serv_addr_));

    if (res < 0)
    {
        fprintf(stderr, "\nCould not bind address to socket");
        return -1;
    }

    fprintf(stderr, "\nBind successful");

    return 0;
}

int TcpServer::check_socket (void)
{
    int       sockoptval = 1;
    socklen_t len        = sizeof(sockoptval);

    if (getsockopt(data_sock_, SOL_SOCKET, SO_ERROR, &sockoptval, &len) < 0)
    {
        fprintf(stderr, "\nError with getsockopt");
        return -1;
    }

    if (sockoptval == 0)
    {
        return 0;
    }

    fprintf(stderr, "\nSocket connection error");
    return -1;
}

int TcpServer::connect_client (void)
{
    int res = listen(sock_, 5);

    if (res < 0)
    {
        fprintf(stderr, "\nSocket listen failed");
        return -1;
    }

    fprintf(stderr, "\nListen successful, waiting for connection...");

    struct sockaddr_in client_addr;

    socklen_t client_addr_size = sizeof(client_addr);
    data_sock_ = accept(sock_, (struct sockaddr *)&client_addr, &client_addr_size);

    if (data_sock_ < 0)
    {
        fprintf(stderr, "\nConnection accept failed [%d]", errno);
        return -1;
    }

    FD_ZERO(&master_);
    FD_SET(data_sock_, &master_);
    fprintf(stderr, "Connection established");

    return 0;
}

int TcpServer::recv_message (void)
{
    struct timeval tv;
    tv.tv_sec = 0;
    tv.tv_usec = 1000;
    readfds_ = master_;
    if (select(data_sock_+1, &readfds_, NULL, NULL, &tv) < 0)
    {
      fprintf(stderr, "\nError with select");
      return -1;
    }
    if (FD_ISSET(data_sock_, &readfds_))
    {
      uint8_t recv_buf;
      int recvlen = recv(data_sock_, &recv_buf, 1, 0);
      if (recvlen < 0) {
        recv_buf = 0;
        return -1;
      }
      //INFO_PRINT("Received: %d", recv_buf);
      return recv_buf;
  }

  return 0;
}

int TcpServer::send_message (const uint8_t* msg, size_t msg_len,
    int msg_id, uint16_t num_cols, uint16_t num_rows, uint8_t* opts,
    int32_t frame, int64_t timestamp)
{
    ImageHeader* msg_hdr = new ImageHeader();
    msg_hdr->message_id = msg_id;
    msg_hdr->flag = 0;
    msg_hdr->frame_id = frame;
    msg_hdr->timestamp_ns = timestamp;
    msg_hdr->num_cols = num_cols;
    msg_hdr->num_rows = num_rows;
    memcpy(msg_hdr->opts, opts, 4*sizeof(uint8_t));
    msg_hdr->checksum = 0; // crc16_init(); ///< @todo do we really need to send checksum?
    msg_hdr->checksum = 0; // crc16(msg_hdr->checksum, msg, msg_len);

    uint8_t *packet = new uint8_t[24];
    memcpy(packet, &msg_hdr->message_id, 1);
    memcpy(packet+1, &msg_hdr->flag, 1);
    memcpy(packet+2, &msg_hdr->frame_id, 4);
    memcpy(packet+6, &msg_hdr->timestamp_ns, 8);
    memcpy(packet+14, &msg_hdr->num_cols, 2);
    memcpy(packet+16, &msg_hdr->num_rows, 2);
    memcpy(packet+18, &msg_hdr->opts, 4);
    memcpy(packet+22, &msg_hdr->checksum, 2);

    // First send the header
    int res = send(data_sock_, packet, 24, 0);
    if (res < 0)
    {
      fprintf(stderr, "\nError in sending image header");
      return -1;
    }

    // Now send the image data
    res = send(data_sock_, msg, msg_len, 0);
    if (res < 0)
    {
      fprintf(stderr, "\nError in sending image");
      return -1;
    }

    delete[] packet;
    delete msg_hdr;

    return 0;
}

int TcpServer::send_message (const uint16_t* msg, size_t msg_len,
    int msg_id, uint16_t num_cols, uint16_t num_rows, uint8_t* opts,
    int32_t frame, int64_t timestamp)
{
    // This function is intended for sending depth map data
    // float is float32_t
    // This message should be 4 times the size of an image
    int m = sizeof(uint16_t)/sizeof(uint8_t);

    ImageHeader* msg_hdr = new ImageHeader();
    msg_hdr->message_id = msg_id;
    msg_hdr->flag = 0;
    msg_hdr->frame_id = frame;
    msg_hdr->timestamp_ns = timestamp;
    msg_hdr->num_cols = num_cols;
    msg_hdr->num_rows = num_rows;
    memcpy(msg_hdr->opts, opts, 4*sizeof(uint8_t));
    msg_hdr->checksum = 0;

    uint8_t *packet = new uint8_t[24];
    memcpy(packet, &msg_hdr->message_id, 1);
    memcpy(packet+1, &msg_hdr->flag, 1);
    memcpy(packet+2, &msg_hdr->frame_id, 4);
    memcpy(packet+6, &msg_hdr->timestamp_ns, 8);
    memcpy(packet+14, &msg_hdr->num_cols, 2);
    memcpy(packet+16, &msg_hdr->num_rows, 2);
    memcpy(packet+18, &msg_hdr->opts, 4);
    memcpy(packet+22, &msg_hdr->checksum, 2);

    int n = num_cols*num_rows;
    for (int i = 0; i < m; ++i)
    {
      // First send the header
      int res = send(data_sock_, packet, 24, 0);
      if (res < 0)
      {
        fprintf(stderr, "\nError in sending image header");
        return -1;
      }

      // Now send the image data
      res = send(data_sock_, msg + (i*(n/m)), msg_len/m, 0);
      if (res < 0)
      {
        fprintf(stderr, "\nError in sending image");
        return -1;
      }
    }

    delete[] packet;
    delete msg_hdr;

    return 0;
}

int TcpServer::send_message (const float* msg, size_t msg_len,
    int msg_id, uint16_t num_cols, uint16_t num_rows, uint8_t* opts,
    int32_t frame, int64_t timestamp)
{
    // This function is intended for sending depth map data
    // float is float32_t
    // This message should be 4 times the size of an image
    int m = sizeof(float)/sizeof(uint8_t);

    ImageHeader* msg_hdr = new ImageHeader();
    msg_hdr->message_id = msg_id;
    msg_hdr->flag = 0;
    msg_hdr->frame_id = frame;
    msg_hdr->timestamp_ns = timestamp;
    msg_hdr->num_cols = num_cols;
    msg_hdr->num_rows = num_rows;
    memcpy(msg_hdr->opts, opts, 4*sizeof(uint8_t));
    msg_hdr->checksum = 0;

    uint8_t *packet = new uint8_t[24];
    memcpy(packet, &msg_hdr->message_id, 1);
    memcpy(packet+1, &msg_hdr->flag, 1);
    memcpy(packet+2, &msg_hdr->frame_id, 4);
    memcpy(packet+6, &msg_hdr->timestamp_ns, 8);
    memcpy(packet+14, &msg_hdr->num_cols, 2);
    memcpy(packet+16, &msg_hdr->num_rows, 2);
    memcpy(packet+18, &msg_hdr->opts, 4);
    memcpy(packet+22, &msg_hdr->checksum, 2);

    int n = num_cols*num_rows;
    for (int i = 0; i < m; ++i)
    {
      // First send the header
      int res = send(data_sock_, packet, 24, 0);
      if (res < 0)
      {
        fprintf(stderr, "\nError in sending image header");
        return -1;
      }

      // Now send the image data
      res = send(data_sock_, msg + (i*(n/m)), msg_len/m, 0);
      if (res < 0)
      {
        fprintf(stderr, "\nError in sending image");
        return -1;
      }
    }

    delete[] packet;
    delete msg_hdr;

    return 0;
}

/** Closes socket
*/
TcpServer::~TcpServer (void)
{
    close(sock_);
    close(data_sock_);
    fprintf(stderr, "\nClosing sockets");
}
