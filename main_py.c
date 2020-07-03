#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <pthread.h>

//#include "htek_sock.h"
#define SVR_DEBUG
#ifdef SVR_DEBUG
	#define SVR_DBG(x,arg...)	printf("[%s][%d]"x,__func__,__LINE__,##arg)
#else
	#define SVR_DBG(x,arg...)
#endif

#define ECHO_SERVER_PORT 7000

typedef struct __client_session{
	int client_sock;
	int session_id;
}cli_session;

int server_sock_handle;
pthread_t g_htekEchoServer = NULL;
pthread_t g_htekClientSession[10] = {NULL, };
int using_client_session[10];

#define CMD_RECV_MOTION_START 	"511-000"
#define CMD_ACK_MOTION_START	"411-000"
#define CMD_RECV_MOTION_END		"511-999"
#define CMD_ACK_MOTION_END		"411-999"

#define CMD_RECV_MOV_START		"522-000"
#define CMD_RECV_FRAME_START	"523-000"
#define CMD_RECV_MOV_END		"533-000"
#define CMD_ACK_MOV_END			"433-000"

typedef enum __LTE_DATA_TYPE{
	LTE_DATA_IDLE = 0,
	LTE_DATA_MOTION_LEN,
	LTE_DATA_MOTION_DATA,
	LTE_DATA_MOV_DATA_LEN,
	LTE_DATA_MOV_DATA,
}LTE_DATA_TYPE;

#define SAVE_DATA

void *client_thread_run(void *args){
	cli_session *l_session = (cli_session *)args;
	int ret;
	unsigned char buffer[8192];
	unsigned char *temp_buffer;
	int read_size;
	int i;
	LTE_DATA_TYPE n_status = LTE_DATA_IDLE;
	unsigned short need_read_size;
	unsigned int mov_need_size;
	unsigned char send_data[11];
	unsigned char py_commend[256] = {0, };
#ifdef SAVE_DATA
	FILE *fWr = 0;
	int data_count = 0;
	unsigned char motion_file_name[256] = {0, };
	unsigned char mov_file_name[256] = {0,};
	struct tm *local_time;
	time_t currentTime;
#endif
	send_data[0] = 0x5c;
	send_data[1] = 0x07;
	send_data[9] = 0xc5;
	//check_sum ����
	
	printf("[%s][%d] run start client ~ \n", __func__, __LINE__);
	while(1){
		read_size = htek_sock_read_timeout(l_session->client_sock, buffer, 8192, 1000);
		if(-1 == read_size){
			printf("[%s][%d] client_sock_id [%d] error ~\n", __func__, __LINE__, l_session->session_id);
			break;
		}else if(0 == read_size){
			continue;
		}
		temp_buffer = buffer;
parsing_start:
		//for(i = 0; i < 11; i++){
		//	SVR_DBG("[%d][0x%x]\n", i, temp_buffer[i]);
		//}
		//printf("[%s][%d] temp_buffer[0] : 0x%x temp_buffer[9] : 0x%x\n", __func__, __LINE__, 
		//		temp_buffer[0], temp_buffer[9]);
		if(temp_buffer[0] == 0x5c && temp_buffer[9] == 0xc5){
			//COMMAND ó�� ��ƾ--------------------------------------------------
			printf("[%s][%d] [%c][%c][%c][%c][%c][%c][%c] \n", __func__, __LINE__,
					temp_buffer[2], temp_buffer[3], temp_buffer[4], temp_buffer[5], 
					temp_buffer[6], temp_buffer[7], temp_buffer[8]);
			if(strncmp(&temp_buffer[2], CMD_RECV_MOTION_START, 7) == 0){
				SVR_DBG("CMD_RECV_MOTION_START \n");
				n_status = LTE_DATA_MOTION_LEN;
				read_size -= 11;
				sprintf(&send_data[2], "%s", CMD_ACK_MOTION_START);
				send_data[9] = 0xc5;
				SVR_DBG(" %s send ~ \n", CMD_ACK_MOTION_START);
#ifdef SAVE_DATA
				if(fWr != 0){
					fclose(fWr);
				}
				memset(motion_file_name, 0x00, sizeof(motion_file_name));
				time(&currentTime);
				local_time = localtime(&currentTime);
				sprintf(motion_file_name, "/home/USERNAME/htekcServer/data/motion/%04d_%02d_%02d_%02d_%02d_%02d",
						local_time->tm_year + 1900, local_time->tm_mon + 1,	(local_time->tm_mday + (local_time->tm_hour+9)/24), 
						(local_time->tm_hour+9)%24, local_time->tm_min, local_time->tm_sec);
				SVR_DBG("motion file name : %s \n", motion_file_name);
				fWr = fopen(motion_file_name, "wb");
#endif
				htek_sock_write_timeout(l_session->client_sock, send_data, 11, 1000);
				if(read_size > 0){
					SVR_DBG("read_size : %d \n", read_size);
					temp_buffer += 11;
					goto parsing_start;
				}
				continue;
			}else if(strncmp(&temp_buffer[2], CMD_RECV_MOTION_END, 7) == 0){
				SVR_DBG("CMD_RECV_MOTION_END ~ \n");
				n_status = LTE_DATA_IDLE;
				read_size -= 11;
				sprintf(&send_data[2], "%s", CMD_ACK_MOTION_END);
				send_data[9] = 0xc5;
				htek_sock_write_timeout(l_session->client_sock, send_data, 11, 1000);
#ifdef SAVE_DATA
				if(fWr != 0){
					SVR_DBG("close motion file ~ \n");
					fclose(fWr);
					fWr = 0;
				}
#endif
				if(read_size > 0){
					SVR_DBG("read_size : %d \n", read_size);
					temp_buffer += 11;
					goto parsing_start;
				}
				continue;
			}else if(strncmp(&temp_buffer[2], CMD_RECV_MOV_START, 7) == 0){
				SVR_DBG("CMD_RECV_MOV_START ~ \n");
				n_status = LTE_DATA_MOV_DATA_LEN;
				read_size -= 11;
#ifdef SAVE_DATA
				if(fWr != 0){
					fclose(fWr);
					fWr = 0;
				}
				memset(mov_file_name, 0x00, sizeof(mov_file_name));
				time(&currentTime);
				local_time = localtime(&currentTime);
				sprintf(mov_file_name, "/home/USERNAME/htekcServer/data/mov/%04d_%02d_%02d_%02d_%02d_%02d.264",
						local_time->tm_year + 1900, local_time->tm_mon + 1,	local_time->tm_mday + (local_time->tm_hour+9)/24, 
						(local_time->tm_hour+9)%24, local_time->tm_min, local_time->tm_sec);
				SVR_DBG("264 file name : %s \n", mov_file_name);
				fWr = fopen(mov_file_name, "wb");
#endif
				if(read_size > 0){
					SVR_DBG("read_size : %d \n", read_size);
					temp_buffer += 11;
					goto parsing_start;
				}
			}else if(strncmp(&temp_buffer[2], CMD_RECV_MOV_END, 7) == 0){
				SVR_DBG("CMD_RECV_MOV_END ~ \n");
				n_status = LTE_DATA_IDLE;
				read_size -= 11;
				sprintf(&send_data[2], "%s", CMD_ACK_MOV_END);
				send_data[9] = 0xc5;
				htek_sock_write_timeout(l_session->client_sock, send_data, 11, 1000);
#ifdef SAVE_DATA
				if(fWr != 0){
					SVR_DBG("close mov file ~ \n");
					fclose(fWr);
					fWr = 0;

					//python code run
					memset(py_commend, 0x00, sizeof(py_commend));
					sprintf(py_commend, "python3 /home/USERNAME/jupyter/videodetect.py --video_path %s", mov_file_name);
					system(py_commend);
				}
#endif
				if(read_size > 0){
					SVR_DBG("read_size : %d \n", read_size);
					temp_buffer += 11;
					goto parsing_start;
				}
				continue;
			}
			//COMMAND ó�� ��ƾ END--------------------------------------------
		//DATA ó�� ��ƾ ------------------------------------------------------
		}else if(n_status == LTE_DATA_MOTION_LEN){
			need_read_size = *(unsigned short *)temp_buffer;
			SVR_DBG("need_read_size : %d \n", need_read_size);
			read_size -= 2;
			n_status = LTE_DATA_MOTION_DATA;
			if(read_size > 0){
				SVR_DBG("read_size : %d \n", read_size);
				temp_buffer += 2;
				goto parsing_start;
			}
			continue;
		}else if(n_status == LTE_DATA_MOTION_DATA){
			SVR_DBG("need_read_size : %d, read_size : %d \n", need_read_size, read_size);
			//�̰����� motion data�� ���� �Ѵ�.
			if(need_read_size == read_size){
				//data �д°��� ���� ����.
				//SVR_DBG("end motion data read\n");
#ifdef SAVE_DATA
				if(fWr != 0){
					fwrite(temp_buffer, 1, read_size, fWr);
					fWr = 0;
				}
#endif
				n_status = LTE_DATA_IDLE;
				read_size = 0;
			}else if(need_read_size > read_size){
				//���� �����Ͱ� ���ִ�.
				//status �� �״�� ����
				need_read_size -= read_size;
				//SVR_DBG("need_read_size : %d \n", need_read_size);
#ifdef SAVE_DATA
				if(fWr != 0){
					fwrite(temp_buffer, 1, read_size, fWr);
				}
#endif
				continue;
			}else if(need_read_size < read_size){
				//data �� �� ������ ���� �����Ͱ� �� �ִ�.
#ifdef SAVE_DATA
				if(fWr != 0){
					fwrite(temp_buffer, 1, need_read_size, fWr);
				}
#endif
				read_size -= need_read_size;
				n_status = LTE_DATA_IDLE;
				temp_buffer += need_read_size;
				//SVR_DBG("restart command read_size : %d\n", read_size);
				goto parsing_start;
			}

		}else if(n_status == LTE_DATA_MOV_DATA_LEN){
			mov_need_size = *(unsigned int *)temp_buffer;
			//SVR_DBG("mov_need_size : %d \n", mov_need_size);
			read_size -= 4;
			n_status = LTE_DATA_MOV_DATA;
			if(read_size > 0){
				//SVR_DBG("read_size : %d \n", read_size);
				temp_buffer += 4;
				goto parsing_start;
			}
			continue;
		}else if(n_status == LTE_DATA_MOV_DATA){
			//SVR_DBG("mov_need_size : %d, read_size : %d \n", mov_need_size, read_size);
			//H.264 Data�� �Ѿ���� �κ��̴�. �̰����� H.264�� ���� �ϸ� �ȴ�.
			if(mov_need_size == read_size){
				//data �д°��� ���� ����.
				//SVR_DBG("end motion data read\n");
#ifdef SAVE_DATA
				if(fWr != 0){
					fwrite(temp_buffer, 1, read_size, fWr);
				}
#endif
				n_status = LTE_DATA_MOV_DATA_LEN;
				read_size = 0;
			}else if(mov_need_size > read_size){
				//���� �����Ͱ� ���ִ�.
				//status �� �״�� ����
				mov_need_size -= read_size;
#ifdef SAVE_DATA
				if(fWr != 0){
					fwrite(temp_buffer, 1, read_size, fWr);
				}
#endif
				//SVR_DBG("mov_need_size : %d \n", mov_need_size);
				continue;
			}else if(mov_need_size < read_size){
				//data �� �� ������ ���� �����Ͱ� �� �ִ�.
#ifdef SAVE_DATA
				if(fWr != 0){
					fwrite(temp_buffer, 1, mov_need_size, fWr);
				}
#endif
				read_size -= mov_need_size;
				n_status = LTE_DATA_MOV_DATA_LEN;
				temp_buffer += mov_need_size;
				//SVR_DBG("restart command read_size : %d\n", read_size);
				goto parsing_start;
			}
		}
		//DATA ó�� ��ƾ ----------------------------------------------------

	}
	using_client_session[l_session->session_id] = 0;
	
	//python code run
//	memset(py_commend, 0x00, sizeof(py_commend));
//	sprintf(py_commend, "python3 /home/USERNAME/jpuyter/videodetect.py --video_path %s", mov_file_name);
//	system(py_commend);

	return 0;
}
void *htek_main_loop(void *args){
	int client_sock = -1;
	struct sockaddr_in client_addr;
	int ret = -1;
	int i;
	cli_session *c_session;

	memset(using_client_session, 0x00, sizeof(using_client_session));
	while(1){
		client_sock = htek_sock_accept_timeout(server_sock_handle, &client_addr, 3000);
		if(-1 == client_sock){
			printf("accept_timeout error \n");
			break;
		}else if(0 == client_sock){
			printf("accep time out ~ \n");
			continue;
		}

		ret = htek_sock_set_non_block(client_sock, 1);
		if(-1 == ret){
			printf("client non block fail ~ \n");
			break;
		}
		for(i = 0; i < 10; i++){
			if(using_client_session[i] == 0){
				using_client_session[i] = 1;
				c_session = (cli_session *)malloc(sizeof(cli_session));
				c_session->session_id = i;
				c_session->client_sock = client_sock;
				printf("[%s][%d] session_id : %d \n", __func__, __LINE__, i);
				ret = pthread_create(&g_htekClientSession[i], NULL, client_thread_run, c_session);
				break;
			}
		}
	}

	return 0;
}

int main(int argc, char *argv[]){
	int ret;
	char buf[32];
	printf("----------------HTEK SERVER TEST----------------\n");
	server_sock_handle = htek_sock_open_tcp_server_socket(ECHO_SERVER_PORT);
	if(-1 == server_sock_handle){
		printf("[%s][%d] sock open fail ~ \n", __func__, __LINE__);
		return 0;
	}
	ret = htek_sock_set_non_block(server_sock_handle, 1);
	pthread_create(&g_htekEchoServer, NULL, htek_main_loop, NULL);
	while(1){
		printf("[%s][%d] e -> exit program ~ \n", __func__, __LINE__);
		gets(buf);
		switch(*buf){
			case 'e':
				printf("[%s][%d] bye~ \n", __func__, __LINE__);
				htek_sock_close(server_sock_handle);
				return 0;
				break;
		}
	}
	htek_sock_close(server_sock_handle);
	return 0;
}
