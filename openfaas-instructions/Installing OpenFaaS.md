#1. INSTALLING OpenFaaS FOR KUBERNETES

Assuming you're using a Mac and MiniKube.

See [Getting started with OpenFaaS on minikube](https://medium.com/devopslinks/getting-started-with-openfaas-on-minikube-634502c7acdf) by Alex Ellis, the originator of OpenFaaS. These steps are taken exactly from the site in the set up on my Mac and worked. Except step 10. I had to rearrange the order of the command arguments for it to work for some reason.

1. Install the OpenFaaS command-line client. 

    `$ brew install faas-cli`

2. Confirm the version number of the CLI installed:

    `$ faas-cli version`

3. Start MiniKube with helm.


##Deploy OpenFaaS to minikube

1. Create a service account for Helm’s server component (tiller): `kubectl -n kube-system create sa tiller && kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller`
2. Install `tiller` which is Helm’s server-side component: `helm init --skip-refresh --upgrade --service-account tiller`
3. Create namespaces for OpenFaaS core components and OpenFaaS Functions: `kubectl apply -f https://raw.githubusercontent.com/openfaas/faas-netes/master/namespaces.yml`
4. Add the OpenFaaS helm repository: `helm repo add openfaas https://openfaas.github.io/faas-netes/`
5. Update all the charts for helm: `helm repo update`
6. Generate a random password: `PASSWORD=$(head -c 12 /dev/urandom | shasum| cut -d' ' -f1)`
7. Create a secret for the password `kubectl -n openfaas create secret generic basic-auth --from-literal=basic-auth-user=admin --from-literal=basic-auth-password="$PASSWORD"`
8. Minikube is not configured for RBAC, so we’ll pass an additional flag to turn it off: `helm upgrade openfaas --install openfaas/openfaas --namespace openfaas --set functionNamespace=openfaas-fn --set basic_auth=true`
9. Set the `OPENFAAS_URL` env-var `export OPENFAAS_URL=$(minikube ip):31112`
10. Finally once all the Pods are started you can login using the CLI: `echo -n $PASSWORD | faas-cli login -u admin —-password-stdin -g http://$OPENFAAS_URL`

You’ll now see the OpenFaaS pods being installed on your minikube cluster. Type in `kubectl get pods -n openfaas` to see them.

## Get the gateway
`export gw=http://$(minikube ip):31112`

`echo $gw`

This is the same as the `OPENFAAS_URL` environment variable set above. Some documentation I've looked at uses `gw` instead. 

## Open the OpenFaaS portal.

To open the web-based portal for OpenFaaS run `open http://$(minikube ip):31112/`

## Docker login
Login to Docker, `$docker login`.

# EXAMPLE FUNCTION DEPLOYMENTS

## a) Deploy a test `hello` function

Again, this is taken directly from [Getting started with OpenFaaS on minikube](https://medium.com/devopslinks/getting-started-with-openfaas-on-minikube-634502c7acdf).

This builds a simple echo function. By default the scaffold template just returns the function argument it's called with. All the files and directories are built automatically in this example.

1. Make a new temporary working directory to build your functions. `$mkdir tmp && cd tmp`.

2. Scaffold a new function

    `faas-cli new --lang python3 hello --prefix="<dockername>" --gateway=$gw`
    
3.    This will create a `hello.yml` file along with a handler folder named `hello` containing your `handler.py` file and `requirements.txt` for any pip modules you may need.

    The `handler.py` file is the actual function code. The scaffolding template simply returns the function arguement, an echo function. 

    > def handle(req):
    > 
    > return(req)

    In general you will put your own function code here.

3. The `hello.yml` file will have set your Docker user name in the `image:` option. The line should look like:

    `image: <docker_name>/hello`
    
    The `gateway:` option will have the Minikube gateway set.
    
    There will also be a `template` directory created to contain templates for different languages.
    
4. In the `tmp` directory invoke a build as:
    `faas-cli build -f hello.yml`
    
    This should then be visible on your local Docker now.
    `docker image ls`
    
5. There will now be a `build` directory in the `tmp` to contain the function builds.

    Note the actual code is contained in the `~/tmp/build/hello/function` hierarchy.

5. Push the versioned Docker image which contains your function up to the Docker Hub. 
`faas-cli push -f hello.yml`

6. Deploy the function.
    `$ faas-cli deploy -f hello.yml --gateway $gw`
    
    Note you can drop the `--gateway` option if you set the gateway to the value of `gw` in the `hello.yml` file.
    
7. Invoke it on the command line.
    `$ echo test | faas-cli invoke hello --gateway $gw`
    
    You can also invoke it using the OpenFaaS web portal opened above.
    
8. List the functions deployed.
`$ faas-cli list --gateway $gw`

9. Delete the `hello` function.
`faas-cli rm hello --gateway=$gw`

## b) Deploy a Multi-file Example Function
    
This is an example of how to (a) import python functions in files separate from the main `handler.py` function and (b) access external files.

1.  Scaffold the function first.
`faas-cli new --lang python3 multi-file --prefix="<dockername>" --gateway=$gw`

2. From git's `openfaas-instructions/multi-file` copy the following files into `~/tmp/multi-file` overwriting any existing prebuilt ones (leave the `__init__.py` alone, it's a python thing):

* `handler.py`
* `requirements.txt`
* `readfile.py`
* `message.txt`

3. Follow the same build and deploy steps as the `hello` function.

* Build: From `~/tmp` run: `faas-cli build -f multi-file.yml`
* Push: `faas-cli push -f multi-file.yml`
* Deploy: `faas-cli deploy -f multi-file.yml --gateway $gw`

You can combine the steps into: `

Alternatively you can do both with `faas-cli up -f ml-test.yml`
This dropped the `--gateway` option since it should have been set in the `faas-cli new ...` command initially.

## c) Deploy a `simple-ml` ML Prediction Example Function

1. In the `tmp` directory run `faas-cli template pull https://github.com/openfaas-incubator/python3-debian` to get the Debian version of python3 to be able to pip install the sklearn, pandas, numpy, etc. modules directly. Don't have to build them from scratch.

2. Scaffold a new function by running:
    `faas-cli new --lang python3-debian simple-ml --prefix="<dockername>" --gateway=$gw`
    Replace `<dockername>` with your docker username.
    
3. As in the `multi-file` example, copy the files in `openfaas-instructions/simple-ml` to the generated `tmp/simple-ml` file, overwriting the templates.

The `requirements.txt` contain the python modules used for the ML operations and will be pip installed by Docker.

The two pickle files, `model.pkl` and `model_columns.pkl` contain the serialized linear regression model to predict survivors of the Titanic.

`handler.py` loads the model, forms the appropriate query structure from the json string sent to the function and returns a 1/0 response on whether the person survived Titanic or not.

The code was based on this [example](https://www.datacamp.com/community/tutorials/machine-learning-models-api-python).

4. Directly build, push and deploy the model.

    `faas-cli up -f simple-ml.yml`

5. In the web portal under the text request body field enter:

>     [{"Age": 85, "Sex": "male", "Embarked": "S"},
>     {"Age": 24, "Sex": "female", "Embarked": "C"},
>     {"Age": 3, "Sex": "male", "Embarked": "C"},
>     {"Age": 21, "Sex": "male", "Embarked": "S"}]

 The response should be: `[0,1,0,0]`. The lady made it.

# 4. NOTES

1. For machine learning in python run
     `faas-cli template pull https://github.com/openfaas-incubator/python3-debian` 
     to get the Debian version of python3 to be able to pip install the sklearn, pandas, numpy, etc. modules directly. You don't have to build them from scratch with Debian.  You do with Alpine and that is a bit more involved although it has a smaller size. 
     
     See https://hub.docker.com/r/frolvlad/alpine-python-machinelearning/dockerfile for an example.
          
2. Use `--prefix="<dockername>"` in the `faas-cli new` command to automatically fill in the proper name for the image (appears in the `<function>.yml` file.

    Similarly use `--gateway=$gw` where `gw` is set by `export gw=http://$(minikube ip):31112`

3. When the portal web page asks for a login set `username` to "admin" and the passord to the value of the environment variable `PASSWORD` set above.

# 5. RESOURCES

1. Main sites are:

    Github: https://github.com/openfaas/faas
    
    Documentation: https://docs.openfaas.com/
    
    Workshop with a series of 11 labs: https://github.com/openfaas/workshop
    
    Katacoda:https://www.katacoda.com/javajon/courses/kubernetes-serverless/openfaas
    
    Slack: The OpenFaas slack channel.