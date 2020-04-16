<?php


namespace led\ModuleQueue\Driver;

use CantExtractPayloadException;
use led\DataProviders\DB;
use led\ModuleQueue\Contract\QueueDriver;
use led\ModuleQueue\Model\Task;
use led\ModuleQueue\QueueLogger;

class DBQueue implements QueueDriver
{
    public const TASK_WAITING_STATUS = 'waiting';
    public const TASK_HANDLING_STATUS = 'handling';
    public const TASK_FINISHED_STATUS = 'finished';

    /**
     * connect queue to db
     * @return bool
     */
    public static function connect(): bool
    {
        return (bool)DB::getInstance();
    }

    /**
     * push delatyed task to db, will handle at configured time
     * @param Task $task
     * @param int $startedAt
     * @return bool
     */
    public static function pushDelayed(Task $task, int $startedAt): bool
    {
        $task->delay($startedAt);

        return static::push($task);
    }

    /**
     * push task to queue
     * @param Task $task
     * @return bool
     */
    public static function push(Task $task): bool
    {
        return static::writeTask($task);
    }

    /**
     * store task in DB
     * @param Task $task
     * @return bool|int
     */
    protected static function writeTask(Task $task)
    {
        $similarTasks = $task->deduplicate ? static::hasSimilarWithStatuses($task, [static::TASK_WAITING_STATUS]) : 0;

        if ($similarTasks > 1 && $task->deduplicate) {
            QueueLogger::emergency('have more when one deduplicated task', ['task' => $task]);
            return true;
        }

        if ($similarTasks > 0 && $task->deduplicate) {
            return true;
        }

        $payload = [
            'pipeline' => $task->pipeline,
            'status' => static::TASK_WAITING_STATUS,
            'data' => json_encode($task->getPayload()),
            'created_at' => time(),
            'handled_at' => 0,
            'started_at' => $task->getStartTime(),
            'handled_times' => 0,
            'deduplicated' => $task->deduplicate,
        ];

        $isInserted = DB::insert('queue_tasks', $payload);

        if (!$isInserted) {
            QueueLogger::emergency('cant save task', [
                'pipeline' => $task->pipeline
            ]);
        }

        QueueLogger::info('task saved', $payload);

        return $isInserted;
    }

    /**
     * check if some similar tasks already exists with some statuses
     * @param Task $task
     * @param array $statuses
     * @return bool
     */
    protected static function hasSimilarWithStatuses(Task $task, array $statuses): bool
    {
        $queryStatuses = array_map(static function ($el) {
            return '\'' . $el . '\'';
        }, $statuses);

        $queryStatuses = implode(',', $queryStatuses);


        $query = "select count(*) from queue_tasks where status in ($queryStatuses) and ";
        $query .= static::similarConditionQuery($task);
        return DB::queryVal($query) > 0;
    }

    /**
     * creates query condition for similar tasks searching in DB
     * @param Task $task
     * @return string
     */
    protected static function similarConditionQuery(Task $task): string
    {
        return sprintf(' id<>\'%s\' and started_at=\'%s\' and pipeline=\'%s\' and `data`=\'%s\'', $task->uuid, $task->getStartTime(), $task->pipeline, json_encode($task->getPayload()));
    }

    /**
     * extract last ready for handling task
     * @return bool|Task|null
     * @throws CantExtractPayloadException
     */
    public static function pop()
    {
        if (static::isEmpty()) {
            return null;
        }

        DB::getInstance()->beginTransaction();

        $taskRow = static::lastActiveTask();

        if (!$taskRow) {
            return null;
        }

        QueueLogger::info('received task', ['id' => $taskRow['id']]);

        $taskLocked = static::lockTask($taskRow);

        if (!$taskLocked) {
            return null;
        }

        QueueLogger::info('task locked', ['id' => $taskRow['id']]);

        $task = Task::createFromDB($taskRow);

        if (!$task) {
            DB::getInstance()->rollBack();
            return false;
        }

        if ($task->deduplicate && static::hasSimilarTaskInRuntime($task)) {
            QueueLogger::emergency('queue have similar task in runtime', ['id' => $task->uuid]);
            DB::getInstance()->rollBack();
            return static::pop();
        }

        DB::getInstance()->commit();

        return $task;
    }

    /**
     * check is queue empty
     * @return bool
     */
    public static function isEmpty(): bool
    {
        return static::length() === 0;
    }

    /**
     * amount of tasks in queue
     * @return int
     */
    public static function length(): int
    {
        return (int)DB::queryVal('select count(*) from queue_tasks where status = ? and started_at <= ?',
            [static::TASK_WAITING_STATUS, time()]);
    }

    /**
     * get last active task from db
     * @return mixed
     */
    protected static function lastActiveTask()
    {
        return DB::queryRow('select * from queue_tasks where status = ? and started_at <= ? 
            order by created_at desc limit 1 for update', [static::TASK_WAITING_STATUS, time()]);
    }

    /**
     * blocking tasks for handling by starting time - if two
     * @param array $task
     * @return bool
     */
    public static function lockTask($task): bool
    {
        $conditions['id'] = $task['id'];

        if ($task['deduplicated']) {
            $conditions = [
                'started_at' => $task['started_at'],
                'data' => $task['data'],
                'pipeline' => $task['pipeline'],
            ];
        }

        return DB::update('queue_tasks', ['status' => static::TASK_HANDLING_STATUS], $conditions);
    }

    /**
     * checks if some similar tasks already work in runtime
     * @param Task $task
     * @return bool
     */
    protected static function hasSimilarTaskInRuntime(Task $task): bool
    {
        return static::hasSimilarWithStatus($task, static::TASK_HANDLING_STATUS) > 0;
    }

    /**
     *  check if some similar tasks already exists with some status
     * @param Task $task
     * @param $status
     * @return bool
     */
    protected static function hasSimilarWithStatus(Task $task, $status): bool
    {

        return DB::queryVal('select count(*) from queue_tasks where status = ? and '
                . static::similarConditionQuery($task), [$status]) > 0;
    }

    /**
     * finalize task after work, if task has error this error will saved
     * @param Task $task
     * @return bool
     */
    public static function finalize(Task $task): bool
    {
        return DB::query('update queue_tasks set handled_times = handled_times + 1, handled_at = ?, status = ?, error = ? where id = ?',
            [time(), static::TASK_FINISHED_STATUS, $task->getError(), $task->uuid])
            ->rowCount();
    }

    /**
     *  check if some similar tasks already exists
     * @param Task $task
     * @return bool
     */
    protected static function hasSimilar(Task $task): bool
    {

        return DB::queryVal('select count(*) from queue_tasks where '
                . static::similarConditionQuery($task)) > 0;
    }

}