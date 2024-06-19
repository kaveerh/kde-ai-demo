#!/bin/bash

echo "AWS_ACCOUNTID  - $AWS_ACCOUNTID"
echo "IOT_BUCKETNAME - $IOT_BUCKETNAME"

QS_ACCOUNT=$(aws quicksight create-account-subscription --edition ENTERPRISE_AND_Q --authentication-method IAM_AND_QUICKSIGHT --aws-account-id "$AWS_ACCOUNTID" --account-name "$IOT_BUCKETNAME" --notification-email "$IOT_BUCKETNAME@me.com" --email-address "$IOT_BUCKETNAME@me.com" --query SignupResponse.userLoginName)
echo "QS_ACCOUNT - $QS_ACCOUNT"
sleep 10

COUNTER=0
QSUSER=$(aws quicksight update-user --role ADMIN_PRO --namespace default --aws-account-id "$AWS_ACCOUNTID" --user-name "WSParticipantRole/Participant" --email "$IOT_BUCKETNAME@me.com" --query User.Arn)

echo $COUNTER - $QSUSER

while [ $COUNTER -lt 5 ] && \
[ "$QSUSER" == "" ]; do
sleep 5
COUNTER=$((COUNTER + 1))
QSUSER=$(aws quicksight update-user --role ADMIN_PRO --namespace default --aws-account-id "$AWS_ACCOUNTID" --user-name "WSParticipantRole/Participant" --email "$IOT_BUCKETNAME@me.com" --query User.Arn)
echo $COUNTER - $QSUSER
done