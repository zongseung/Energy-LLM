### NAS 마운트 방법
  - sudo mount -t cifs //ip/naverResearch /mnt/data_sdd/energy_gpt/nas-mount/naverDB \
-o username=사용자명,password='비밀번호',vers=3.0,uid=$(id -u),gid=$(id -g),file_mode=0777,dir_mode=0777| 옵션               | 의미                                                |

이후 코드에 대한 설명은 다음과 같다.
| ---------------- | ------------------------------------------------- |
| `uid=$(id -u)`   | 마운트 시 파일의 \*\*소유자(user)\*\*를 현재 로그인한 사용자의 UID로 설정 |
| `gid=$(id -g)`   | 마운트 시 파일의 \*\*그룹(group)\*\*을 현재 사용자의 GID로 설정      |
| `file_mode=0777` | 마운트된 NAS의 **모든 파일에 읽기/쓰기/실행 권한**(rwxrwxrwx) 부여    |
| `dir_mode=0777`  | 마운트된 NAS의 **모든 디렉토리에 읽기/쓰기/실행 권한** 부여             |

  - `모든 작업은 sudo unmount <NAS 주소>`으로 하고 추가 작업을 진행한다. 안그러면 NAS가 인식을 잘못하여 차단 IP 로 설정해버린다.
  - 
