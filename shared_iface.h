extern "C" {
int getMemHandleSize();
char* getMemHandle(void *mem);
int openMemHandle(void **mem, char *handle_str);
int closeMemHandle(void *mem);
}
