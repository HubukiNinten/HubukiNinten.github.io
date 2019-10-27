#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>


using namespace std;

class node	//연결리스트 
{
public:
	int num;				//좌석번호
	char name[11];			//이름 (최대 10글자)
	node* next;//다음 주소를 저장할 포인터변수
};

class headlist {				//헤드
public:
	node* link;				//노드 연결할 포인터변수
};

void createlist(headlist* L, int i) // 120번까지의 자리 노드 만들기
{
	node* newnode;
	node* temp;
	newnode = new node;
	newnode->num = i;
	newnode->name[0] = 0;
	newnode->next = NULL;
	if (L->link == NULL)
	{
		L->link = newnode;
		return;
	}
	temp = L->link;
	while (temp->next != NULL)
		temp = temp->next;
	temp->next = newnode;
}

void seat(headlist* L) 
{
	node* temp;
	char arr[125];
	temp = L->link;
	if (L->link->name[0] != 0)	
		arr[0] = 'O';
	else

		arr[0] = 'X';
	for (int i = 1; i < 120; i++)
	{
		while (temp->num != i + 1)
			temp = temp->next;
		if (temp->name[0] != 0)
			arr[i] = 'O';
		else
			arr[i] = 'X';
	}
	int u, k = 0, l = 0;			
	u = (10 * k) + l;
	cout<<"   행	/1	2	3	4	5	6	7	8	9	10"<<endl; 
	for (k = 0; k < 12; k++)
	{
		cout << k << "열	/";
		for (l = 0; l < 10; l++)
		{
			u = (10 * k) + l;
			cout<<arr[u] << "	  ";
		}
		cout<<endl<<endl;
	}
}
void enter(headlist* L, char* namet, int i)
{
	node* temp;
	temp = L->link;
	while (temp->num != i)
		temp = temp->next;					//자리 번호가 저장된 노드로 이동
	strncpy_s(temp->name, namet, sizeof(temp->name));	//지정된 자리에 이름 등록

}

void out(headlist* L, int i) 
{
	node* temp;
	temp = L->link;
	while (temp->num != i)
		temp = temp->next;					
	temp->name[0] = 0; 
}

int checkover(headlist* L, int i) 
{
	node* temp;
	temp = L->link;
	while (temp->num != i)
		temp = temp->next;//자리번호가 저장된 노드로 이동
	if (temp->name[0] != 0)//자리에 이름이 비어있지 않을시
		return 1;		// 1 반환
	else
		return 0;		// 이름이 비어있으면 0 반환
}


int main()
{
	headlist* L; //헤드 선언
	L = (headlist*)new(headlist);//헤드에 동적 메모리 할당
	L->link = NULL;
	for (int i = 1; i <= 120; i++)
		createlist(L, i);//연결리스트 생성

numch: //초기화면
	cout<<"-----------------------------------------------------------------"<<endl;
	cout<<"//  무인 주차장 관리 프로그램을 이용해 주셔서 감사합니다.      //"<<endl;
	cout<<"\n\n\n";
	cout<<"// 원하시는 기능에 해당하는 번호를 입력 후 Enter를 눌러주세요. //\n";
	cout<<endl;
	cout<<"// 기능 : ";
	cout<<"/ 1.   주차    / 2.  차 빼기  / 3. 프로그램종료      //\n";
	cout<<"-----------------------------------------------------------------\n";
	int num;//자리 번호를 입력받음
	char stop;//Y또는N을 입력받음
	char namet[11];//이름을 입력받음
	cout<<"숫자를 입력하세요 : ";
	cin>>num;
	while (getchar() != '\n');
	if (num == 1) 
	{
		seat(L);
	No_1:
		cout<<"자리 번호를 선택해주세요 : ";								//입실할 자리 번호 선택
		cin>>num;
		if ((num < 1) || (120 < num))							//1~120사이의 숫자가 아니거나 문자열일 경우
		{
			cout<<"번호를 다시 입력해주세요.\n";
			while (getchar() != '\n');
			goto No_1;
		}
		int check = checkover(L, num);										//선택한 자리에 누가 있는지 검색
		if (check == 1)																//사람이 이미 있을 경우
		{
			cout<<"선택하신 자리에 차가 이미 있습니다.\n다시 선택해주세요.\n\n";
			goto No_1;
		}
		cout<<"이름을 입력해주세요 : ";							//선택한곳에 아무도 없을 경우 이름 입력
		cin>>namet;
		while (getchar() != '\n');
	re_1:
		cout<<namet<<"님, "<<num<<"번 자리가 맞습니까? Y/N :";					//사람이 없을 경우 재확인
		stop = getchar();
		while (getchar() != '\n');
		if (stop == 'Y')
		{
			enter(L, namet, num);
			cout<<"주차 완료 되었습니다.\n";
			goto numch;												//초기 선택 화면으로 돌아감
		}
		else if (stop == 'N')										//아닐 경우 다시 입실할 번호를 선택하러 돌아감
			goto No_1;
		else
		{													//입력 받은게 Y 또는 N이 아닌 경우 다시 입력
			cout<<"다시 입력해 주십시오.\n";
			while (getchar() != '\n');
			goto re_1;
		}
	}
	else if (num == 2) //자리 퇴실
	{
	No_2:
		seat(L);											//자리 현황 보여줌
		cout<<"자리 번호를 선택해주세요 : ";				//퇴실할 자리 번호 선택
		cin>>num;
		if ((num < 1) || (120 < num))							//1~120사이의 숫자가 아니거나 문자열일 경우
		{
			cout<<"번호를 다시 입력해주세요.\n";
			while (getchar() != '\n');
			goto No_2;
		}
		while (getchar() != '\n');
	re_2:
		cout << num << "번 자리가 맞습니까? Y/N :";			// 번호 재확인
		stop = getchar();
		while (getchar() != '\n');
		
		if (stop == 'Y')
		{
			out(L, num);
			cout<<"발차 처리 되었습니다.\n";
			goto numch;										//초기 화면으로 되돌아감
		}
		else if (stop == 'N')								//퇴실할 번호 잘못 선택시 다시 돌아감
			goto No_2;
		else
		{													//Y 또는 N이 아닌 경우 다시 입력받음
			cout<<"다시 입력해 주십시오.\n";
			goto re_2;
		}
	}
	else if (num == 3) //프로그램 종료
	{
	end:
		cout<<"프로그램을 종료하시겠습니까?  Y/N : ";
		stop = getchar();
		while (getchar() != '\n');
		
		if (stop == 'Y')									// Y 입력 받으면 종료
		{
			cout<<"이용 해 주셔서 감사합니다.\n";
			system("pause");
			return 0;
		}
		else if (stop == 'N')								// N 입력 받으면초기 화면으로 되돌아감
			goto numch;
		else
		{
			cout<<"다시 입력해 주십시오.\n";
			goto end;
		}
	}
	else													// 1,2,3이 아닌 것을 입력 받으면 초기화면으로
	{
		cout<<"다시 입력해 주십시오.\n";
		goto numch;
	}
	//프로그램 종료
	system("pause");
	return 0;
}