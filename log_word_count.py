from collections import Counter
f = open('log.log', 'r')
r = f.readlines()
datestr=[]

for i in range(len(r)):
    datestr.append(r[i][:12])

dates = Counter(datestr)

dic= {}
for k , v in dates.items():
    dic.update({k: {'total_count': v, 'logrotate' : 0, 'run-parts': 0, 'anacron':0,'CROND':0,'ntpd':0, 'rsyslogd':0, 'cs3':0,'ACCT_ADD':0}})

for k , v in dic.items():
    
    for i in range(len(r)):
        if k == r[i][:12] :
            v['logrotate']+=r[i].count('logrotate')
            v['run-parts']+=r[i].count('run-parts')
            v['anacron']+=r[i].count('anacron')
            v['CROND']+=r[i].count('CROND')
            v['ntpd']+=r[i].count('ntpd')
            v['rsyslogd']+=r[i].count('rsyslogd')
            v['ACCT_ADD']+=r[i].count('ACCT_ADD')
            v['cs3']+=r[i].count('cs3')
            
print(dic)

print("minute,total_count,logrotate,run-parts,anacron,CROND,ntpd,rsyslogd,cs3,ACCT_ADD")
for k , v in dic.items():
    print(k,v['total_count'], v['logrotate'],v['run-parts'],v['anacron'],v['CROND'],v['ntpd'],v['rsyslogd'],v['cs3'],v['ACCT_ADD'])
