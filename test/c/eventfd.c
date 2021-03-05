//gcc eventfd.c -o eventfd -lpthread
#include <stdio.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/eventfd.h>
#include <sys/epoll.h>

int efd = -1;

void *read_thread(void *dummy)
{
    int ret = 0;
    uint64_t count = 0;
    int ep_fd = -1;
    struct epoll_event events[10];

    if (efd < 0)
    {
        printf("efd not inited.\n");
        goto fail;
    }

    ep_fd = epoll_create(1024);
    if (ep_fd < 0)
    {
        perror("epoll_create fail: ");
        goto fail;
    }


    struct epoll_event read_event;

    read_event.events = EPOLLHUP | EPOLLERR | EPOLLIN;
    read_event.data.fd = efd;

    ret = epoll_ctl(ep_fd, EPOLL_CTL_ADD, efd, &read_event);
    if (ret < 0)
    {
        perror("epoll ctl failed:");
        goto fail;
    }


    while (1)
    {
        ret = epoll_wait(ep_fd, &events[0], 10, 5000);
        if (ret > 0)
        {
            int i = 0;
            for (; i < ret; i++)
            {
                if (events[i].events & EPOLLHUP)
                {
                    printf("epoll eventfd has epoll hup.\n");
                    goto fail;
                }
                else if (events[i].events & EPOLLERR)
                {
                    printf("epoll eventfd has epoll error.\n");
                    goto fail;
                }
                else if (events[i].events & EPOLLIN)
                {
                    int event_fd = events[i].data.fd;
                    ret = read(event_fd, &count, sizeof(count));
                    if (ret < 0)
                    {
                        perror("read fail:");
                        goto fail;
                    }
                    else
                    {
                        struct timeval tv;

                        gettimeofday(&tv, NULL);
                        printf("success read from efd, read %d bytes(%lu) at %lds %ldus\n",
                               ret, count, tv.tv_sec, tv.tv_usec);
                    }
                }
            }
        }
        else if (ret == 0)
        {
            /* time out */
            printf("epoll wait timed out.\n");
            break;
        }
        else
        {
            printf("ret is %d \n", ret);
            perror("epoll wait error:");
            goto fail;
        }
    }

fail:
    if (ep_fd >= 0)
    {
        close(ep_fd);
        ep_fd = -1;
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    pthread_t pid = 0;
    uint64_t count = 0;
    int ret = 0;
    int i = 0;

    //efd = eventfd(0, 0);
    printf("EFD_NONBLOCK:%d\n",EFD_NONBLOCK);
    efd = eventfd(0, EFD_NONBLOCK);
    if (efd < 0)
    {
        perror("eventfd failed.");
        goto fail;
    }

    ret = pthread_create(&pid, NULL, read_thread, NULL);
    if (ret < 0)
    {
        perror("pthread create:");
        goto fail;
    }

    for (i = 0; i < 5; i++)
    {
        count = 4;
        ret = write(efd, &count, sizeof(count));
        if (ret < 0)
        {
            perror("write event fd fail:");
            goto fail;
        }
        else
        {
            struct timeval tv;

            gettimeofday(&tv, NULL);
            printf("success write to efd, write %d bytes(%lu) at %lds %ldus\n",
                   ret, count, tv.tv_sec, tv.tv_usec);
        }

        sleep(1);
    }

fail:
    if (0 != pid)
    {
        pthread_join(pid, NULL);
        pid = 0;
    }

    if (efd >= 0)
    {
        close(efd);
        efd = -1;
    }
    return ret;
}