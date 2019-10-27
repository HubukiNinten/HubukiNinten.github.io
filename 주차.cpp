#include <iostream>
#include <cstdlib>
#include <cstring>
#include <string>


using namespace std;

class node	//���Ḯ��Ʈ 
{
public:
	int num;				//�¼���ȣ
	char name[11];			//�̸� (�ִ� 10����)
	node* next;//���� �ּҸ� ������ �����ͺ���
};

class headlist {				//���
public:
	node* link;				//��� ������ �����ͺ���
};

void createlist(headlist* L, int i) // 120�������� �ڸ� ��� �����
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
	cout<<"   ��	/1	2	3	4	5	6	7	8	9	10"<<endl; 
	for (k = 0; k < 12; k++)
	{
		cout << k << "��	/";
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
		temp = temp->next;					//�ڸ� ��ȣ�� ����� ���� �̵�
	strncpy_s(temp->name, namet, sizeof(temp->name));	//������ �ڸ��� �̸� ���

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
		temp = temp->next;//�ڸ���ȣ�� ����� ���� �̵�
	if (temp->name[0] != 0)//�ڸ��� �̸��� ������� ������
		return 1;		// 1 ��ȯ
	else
		return 0;		// �̸��� ��������� 0 ��ȯ
}


int main()
{
	headlist* L; //��� ����
	L = (headlist*)new(headlist);//��忡 ���� �޸� �Ҵ�
	L->link = NULL;
	for (int i = 1; i <= 120; i++)
		createlist(L, i);//���Ḯ��Ʈ ����

numch: //�ʱ�ȭ��
	cout<<"-----------------------------------------------------------------"<<endl;
	cout<<"//  ���� ������ ���� ���α׷��� �̿��� �ּż� �����մϴ�.      //"<<endl;
	cout<<"\n\n\n";
	cout<<"// ���Ͻô� ��ɿ� �ش��ϴ� ��ȣ�� �Է� �� Enter�� �����ּ���. //\n";
	cout<<endl;
	cout<<"// ��� : ";
	cout<<"/ 1.   ����    / 2.  �� ����  / 3. ���α׷�����      //\n";
	cout<<"-----------------------------------------------------------------\n";
	int num;//�ڸ� ��ȣ�� �Է¹���
	char stop;//Y�Ǵ�N�� �Է¹���
	char namet[11];//�̸��� �Է¹���
	cout<<"���ڸ� �Է��ϼ��� : ";
	cin>>num;
	while (getchar() != '\n');
	if (num == 1) 
	{
		seat(L);
	No_1:
		cout<<"�ڸ� ��ȣ�� �������ּ��� : ";								//�Խ��� �ڸ� ��ȣ ����
		cin>>num;
		if ((num < 1) || (120 < num))							//1~120������ ���ڰ� �ƴϰų� ���ڿ��� ���
		{
			cout<<"��ȣ�� �ٽ� �Է����ּ���.\n";
			while (getchar() != '\n');
			goto No_1;
		}
		int check = checkover(L, num);										//������ �ڸ��� ���� �ִ��� �˻�
		if (check == 1)																//����� �̹� ���� ���
		{
			cout<<"�����Ͻ� �ڸ��� ���� �̹� �ֽ��ϴ�.\n�ٽ� �������ּ���.\n\n";
			goto No_1;
		}
		cout<<"�̸��� �Է����ּ��� : ";							//�����Ѱ��� �ƹ��� ���� ��� �̸� �Է�
		cin>>namet;
		while (getchar() != '\n');
	re_1:
		cout<<namet<<"��, "<<num<<"�� �ڸ��� �½��ϱ�? Y/N :";					//����� ���� ��� ��Ȯ��
		stop = getchar();
		while (getchar() != '\n');
		if (stop == 'Y')
		{
			enter(L, namet, num);
			cout<<"���� �Ϸ� �Ǿ����ϴ�.\n";
			goto numch;												//�ʱ� ���� ȭ������ ���ư�
		}
		else if (stop == 'N')										//�ƴ� ��� �ٽ� �Խ��� ��ȣ�� �����Ϸ� ���ư�
			goto No_1;
		else
		{													//�Է� ������ Y �Ǵ� N�� �ƴ� ��� �ٽ� �Է�
			cout<<"�ٽ� �Է��� �ֽʽÿ�.\n";
			while (getchar() != '\n');
			goto re_1;
		}
	}
	else if (num == 2) //�ڸ� ���
	{
	No_2:
		seat(L);											//�ڸ� ��Ȳ ������
		cout<<"�ڸ� ��ȣ�� �������ּ��� : ";				//����� �ڸ� ��ȣ ����
		cin>>num;
		if ((num < 1) || (120 < num))							//1~120������ ���ڰ� �ƴϰų� ���ڿ��� ���
		{
			cout<<"��ȣ�� �ٽ� �Է����ּ���.\n";
			while (getchar() != '\n');
			goto No_2;
		}
		while (getchar() != '\n');
	re_2:
		cout << num << "�� �ڸ��� �½��ϱ�? Y/N :";			// ��ȣ ��Ȯ��
		stop = getchar();
		while (getchar() != '\n');
		
		if (stop == 'Y')
		{
			out(L, num);
			cout<<"���� ó�� �Ǿ����ϴ�.\n";
			goto numch;										//�ʱ� ȭ������ �ǵ��ư�
		}
		else if (stop == 'N')								//����� ��ȣ �߸� ���ý� �ٽ� ���ư�
			goto No_2;
		else
		{													//Y �Ǵ� N�� �ƴ� ��� �ٽ� �Է¹���
			cout<<"�ٽ� �Է��� �ֽʽÿ�.\n";
			goto re_2;
		}
	}
	else if (num == 3) //���α׷� ����
	{
	end:
		cout<<"���α׷��� �����Ͻðڽ��ϱ�?  Y/N : ";
		stop = getchar();
		while (getchar() != '\n');
		
		if (stop == 'Y')									// Y �Է� ������ ����
		{
			cout<<"�̿� �� �ּż� �����մϴ�.\n";
			system("pause");
			return 0;
		}
		else if (stop == 'N')								// N �Է� �������ʱ� ȭ������ �ǵ��ư�
			goto numch;
		else
		{
			cout<<"�ٽ� �Է��� �ֽʽÿ�.\n";
			goto end;
		}
	}
	else													// 1,2,3�� �ƴ� ���� �Է� ������ �ʱ�ȭ������
	{
		cout<<"�ٽ� �Է��� �ֽʽÿ�.\n";
		goto numch;
	}
	//���α׷� ����
	system("pause");
	return 0;
}