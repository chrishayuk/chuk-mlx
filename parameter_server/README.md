

## LLM Simulator Client
So i created a very rough simulator client, that performs a simulation of sending some updates from a training node to the parameter server.  This all runs locally.  In the test, i run 2 tabs in terminal, configured with 10 clients running sending updates to the parameter server.  These are not sending real training weights updates, but is just sending numbers to see what the server can cope with.

```bash
python llm_send_client.py  --client_count 10 --batch_size 128 --sequence_length 256 --num_updates 200 --update_interval 0.05
```

in this case, the server could easily cope with 17 updates per second per client (or 571,000 tokens per second per client)

```bash
Connected to server at localhost:5555
Client ID: 484e02b4-30a4-420c-b130-818f7d196c0c
Start Time: 2024-08-20 00:52:08
[484e02b4-30a4-420c-b130-818f7d196c0c] Updates Sent: 200/200 | Updates Per Second: 17.35 | Tokens Per Second: 568427.93 | Current Version: 6570
Connected to server at localhost:5555
Client ID: 8496e9cc-a031-4899-9438-517402350f54
Start Time: 2024-08-20 00:52:08
[8496e9cc-a031-4899-9438-517402350f54] Updates Sent: 200/200 | Updates Per Second: 17.27 | Tokens Per Second: 565795.87 | Current Version: 6573
Connected to server at localhost:5555
Client ID: 46af0163-6801-4e4b-9572-3e4383a1baac
Start Time: 2024-08-20 00:52:08
[46af0163-6801-4e4b-9572-3e4383a1baac] Updates Sent: 200/200 | Updates Per Second: 17.44 | Tokens Per Second: 571516.48 | Current Version: 6568
Connected to server at localhost:5555
Client ID: 8ba97813-861a-4613-974d-a89ca37329f0
Start Time: 2024-08-20 00:52:08
[8ba97813-861a-4613-974d-a89ca37329f0] Updates Sent: 200/200 | Updates Per Second: 17.28 | Tokens Per Second: 566264.66 | Current Version: 6573
Connected to server at localhost:5555
Client ID: 7b6a00c0-f5bc-454f-81fc-32499fd3f13e
Start Time: 2024-08-20 00:52:08
[7b6a00c0-f5bc-454f-81fc-32499fd3f13e] Updates Sent: 200/200 | Updates Per Second: 17.29 | Tokens Per Second: 566655.13 | Current Version: 6572
Connected to server at localhost:5555
Client ID: af8ff6b9-fa3f-431c-9dde-50e1ae0ca459
Start Time: 2024-08-20 00:52:08
[af8ff6b9-fa3f-431c-9dde-50e1ae0ca459] Updates Sent: 200/200 | Updates Per Second: 17.31 | Tokens Per Second: 567112.72 | Current Version: 6574
Connected to server at localhost:5555
Client ID: 138ab8cd-eec1-448c-9f17-eb67f8c86b08
Start Time: 2024-08-20 00:52:08
[138ab8cd-eec1-448c-9f17-eb67f8c86b08] Updates Sent: 200/200 | Updates Per Second: 17.29 | Tokens Per Second: 566641.36 | Current Version: 6572
Connected to server at localhost:5555
Client ID: d81cfd89-29fe-47af-9d9f-646a9cceb6c9
Start Time: 2024-08-20 00:52:08
[d81cfd89-29fe-47af-9d9f-646a9cceb6c9] Updates Sent: 200/200 | Updates Per Second: 17.31 | Tokens Per Second: 567353.10 | Current Version: 6572
Connected to server at localhost:5555
Client ID: 6b8bbc48-644a-4719-9966-4513805169c1
Start Time: 2024-08-20 00:52:08
[6b8bbc48-644a-4719-9966-4513805169c1] Updates Sent: 200/200 | Updates Per Second: 17.31 | Tokens Per Second: 567368.57 | Current Version: 6572
Connected to server at localhost:5555
Client ID: 8e8a6aea-d779-4203-9a44-d1bfecd4a5ba
Start Time: 2024-08-20 00:52:08
[8e8a6aea-d779-4203-9a44-d1bfecd4a5ba] Updates Sent: 200/200 | Updates Per Second: 17.41 | Tokens Per Second: 570578.73 | Current Version: 6571
```

if these numbers hold true then the parameter server should be perfectly capable and adequate of updating parameters at this rate.

